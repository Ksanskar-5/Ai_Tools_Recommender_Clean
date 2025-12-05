"""
Hybrid recommendation service.

Now supports:
- Explicit feedback (likes/dislikes)
- Implicit feedback (search/view history)
- Collaborative filtering (similar users)       
- Content-based similarity using USER PROFILE ONLY
"""

import logging
from typing import List, Dict, Any

from backend.utils.db import (
    get_user_feedback,
    get_user_actions,
    get_all_ai,
    get_similar_users_feedback,
)

from backend.services.recommender import (

    normalize_records,
    llm_fallback,
)

from backend.utils.vectorizer import semantic_search as vs_semantic_search

logger = logging.getLogger(__name__)
# -------------------------------
# Hybrid Weights (base)
# -------------------------------
# Base weights (will be adapted per-user)
WEIGHTS = {
    "content": 0.45,   # content/personal
    "collab": 0.30,    # collaborative
    "semantic": 0.15,  # semantic / vector match
    "engagement": 0.10 # implicit engagement (CTR, dwell, visits)
}

 
# ============================================================
# 1. Personalized Content Scoring (using user profile)
# ============================================================
def get_user_personal_scores(user_id: int, top_k: int = 50) -> Dict[int, float]:
    """
    Build a per-item personal score using:
      - explicit feedback (1 or -1) from feedback table
      - rating (1-5) normalized to [0,1]
      - bookmark existence (binary)
      - comment sentiment (-1..1) normalized to [0,1]
    Returns: {ai_id: score_in_range[-1..1]} (clipped)
    """
    if not user_id:
        return {}

    from backend.utils.db import get_connection

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # Query explicit fields from feedback table and bookmarks
    cur.execute("""
        SELECT f.ai_id,
               f.feedback,
               f.rating,
               f.comment,
               f.sentiment,
               (CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END) AS bookmarked
        FROM feedback f
        LEFT JOIN bookmarks b ON f.user_id = b.user_id AND f.ai_id = b.ai_id
        WHERE f.user_id = %s
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    scores = {}
    # weights for components contributing to personal score
    w_feedback = 1.0      # like/dislike (±1)
    w_rating = 0.8        # rating normalized 0..1
    w_bookmark = 1.0      # bookmark bonus 0/1
    w_sentiment = 0.5     # comment sentiment scaled 0..1

    max_possible = w_feedback + w_rating + w_bookmark + w_sentiment  # used for normalization

    for r in rows:
        ai_id = r["ai_id"]
        comp = 0.0

        # explicit like/dislike: already stored as 1 or -1 (may be None)
        fb = r.get("feedback")
        if fb is not None:
            comp += w_feedback * float(fb)   # keeps sign

        # rating -> normalized 0..1
        rating = r.get("rating")
        if rating is not None:
            rating_norm = max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))
            comp += w_rating * rating_norm

        # bookmark
        if r.get("bookmarked"):
            comp += w_bookmark * 1.0

        # sentiment (-1..1 -> 0..1)
        sent = r.get("sentiment")
        if sent is not None:
            sent_norm = (float(sent) + 1.0) / 2.0
            comp += w_sentiment * sent_norm

        # normalize to approx [-1, +1] range and clip
        # when fb exists and is -1, comp could be negative; normalize by max_possible
        norm_score = comp / max_possible
        # if only a negative feedback exists (fb=-1), ensure it's reflected strongly
        if fb == -1:
            norm_score = max(norm_score, -0.95)

        scores[ai_id] = max(-1.0, min(1.0, norm_score))

    return scores


# ============================================================
# 2 . Collaborative Filtering
# ============================================================
def get_collaborative_scores(user_id: int, ai_data: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Collaborative score computed from aggregated explicit feedback across users.
    Score is like_ratio in [0..1] (higher means more liked by crowd).
    """
    from backend.utils.db import get_connection

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # Aggregate likes/dislikes from feedback table (preferred source for explicit signals)
    cur.execute("""
        SELECT ai_id,
               SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS likes,
               SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) AS dislikes
        FROM feedback
        GROUP BY ai_id;
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    scores = {}
    for r in rows:
        ai_id = r["ai_id"]
        likes = r.get("likes") or 0
        dislikes = r.get("dislikes") or 0
        total = likes + dislikes
        if total == 0:
            score = 0.0
        else:
            # like ratio in [0..1]
            score = likes / total
        scores[ai_id] = float(score)
    return scores



def compute_engagement_scores(user_id: int, top_k: int = 50) -> Dict[int, float]:
    """
    Compute engagement-based scores per ai_id from user_data:
      - CTR (clicks / impressions) smoothed
      - avg dwell time normalized
      - repeat visits -> log-scaled
      - search_refinements -> negative signal
    Returns normalized score in [0..1] per ai_id.
    """
    from backend.utils.db import get_connection

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # Fetch aggregated events for this user across ai items
    cur.execute("""
        SELECT ai_id,
               SUM(action = 'impression') AS impressions,
               SUM(action = 'click') AS clicks,
               SUM(action = 'view') AS views,
               SUM(action = 'visit') AS visits,
               SUM(action = 'search_refine') AS refinements,
               AVG(COALESCE(dwell_time,0)) AS avg_dwell
        FROM user_data
        WHERE user_id = %s AND ai_id IS NOT NULL
        GROUP BY ai_id;
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    engagement = {}
    # smoothing constant
    alpha = 3.0

    for r in rows:
        ai_id = r["ai_id"]
        impressions = r.get("impressions") or 0
        clicks = r.get("clicks") or 0
        avg_dwell = float(r.get("avg_dwell") or 0.0)
        visits = r.get("visits") or 0
        refinements = r.get("refinements") or 0

        # smoothed CTR
        ctr = (clicks + 0.5) / (impressions + alpha)

        # dwell normalized (cap at 300s)
        dwell_norm = min(avg_dwell, 300.0) / 300.0

        # visits -> log scale
        visit_score = 0.0
        if visits > 0:
            import math
            visit_score = math.log(1 + visits)

        # refinements reduce score (normalize)
        refine_penalty = min(refinements / 5.0, 1.0)

        # combine into a single engagement score (0..1)
        score = (0.45 * ctr) + (0.35 * dwell_norm) + (0.15 * (visit_score / (1 + visit_score))) - (0.05 * refine_penalty)
        score = max(0.0, min(1.0, score))
        engagement[ai_id] = score

    return engagement


# ============================================================
# 4. Hybrid Recommendations (Query + Personalization)
# ============================================================
def get_hybrid_recommendations(query: str, user_id: int = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Combine semantic candidates with:
      - personalized content (explicit)
      - collaborative (crowd)
      - engagement (implicit)
    Uses adaptive weighting depending on user's explicit signal count.
    """
    try:
        ai_data = get_all_ai()
        if not ai_data:
            return normalize_records(llm_fallback(query))

        # --- Signals ---
        personal_scores = get_user_personal_scores(user_id, top_k=top_k) if user_id else {}
        collab_scores = get_collaborative_scores(user_id, ai_data) if user_id else {}
        engagement_scores = compute_engagement_scores(user_id, top_k=top_k) if user_id else {}

        # Semantic search results (fall back to LLM if needed)
        try:
            semantic_results = vs_semantic_search(query, top_k=top_k)
            semantic_scores = {
                r.get("ai_id") or r.get("ID"): 1.0 - (i / float(max(1, top_k)))
                for i, r in enumerate(semantic_results)
            }
        except Exception:
            semantic_scores = {}

        # Determine adaptive adjustment based on how much explicit data the user has
        explicit_count = len(personal_scores)
        # If user has very few explicit signals, prefer content (personal/CBF),
        # otherwise favor collaborative.
        if explicit_count < 5:
            adapt_mul = {"content": 1.2, "collab": 0.8}
        else:
            adapt_mul = {"content": 0.9, "collab": 1.1}

        # start from base weights and apply adaptation, then normalize to sum=1
        base = dict(WEIGHTS)
        base["content"] *= adapt_mul.get("content", 1.0)
        base["collab"] *= adapt_mul.get("collab", 1.0)
        # keep engagement and semantic as-is
        total = sum(base.values())
        weights = {k: v / total for k, v in base.items()}

        # --- Merge signals per ai_id ---
        hybrid_scores = {}
        all_ai_ids = set(personal_scores) | set(collab_scores) | set(semantic_scores) | set(engagement_scores)

        for ai_id in all_ai_ids:
            p = personal_scores.get(ai_id, 0.0)
            cf = collab_scores.get(ai_id, 0.0)
            s = semantic_scores.get(ai_id, 0.0)
            e = engagement_scores.get(ai_id, 0.0)

            final = (
                weights["content"] * p +
                weights["collab"] * cf +
                weights["semantic"] * s +
                weights["engagement"] * e
            )

            # small boost: if bookmarked, increase score slightly (read from bookmarks table)
            # we attempt to detect bookmark by querying feedback/bookmarks quickly
            hybrid_scores[ai_id] = float(final)

        # --- Sort & take top ---
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        ai_map = {a["ai_id"]: a for a in ai_data}
        results = []
        for ai_id, score in ranked:
            if ai_id in ai_map:
                rec = dict(ai_map[ai_id])
                rec["score"] = float(score)
                rec["Reasoning"] = "Hybrid personalized ranking (content+collab+semantic+engagement)"
                results.append(rec)

        return normalize_records(results)

    except Exception as e:
        logger.error("Hybrid recommendation failed: %s", e)
        return normalize_records([{
            "ai_id": 0,
            "Name": "Hybrid-Error",
            "Category": "Error",
            "Task Description": str(e),
            "score": 0.0
        }])

def apply_filters(candidate_map: Dict[int, Dict[str, Any]], filters: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Simple filter logic applied to the ai_data map before scoring.
    Supported filters example:
      - category: 'Image'
      - company: 'OpenAI'
      - cost: 'Free'
    Returns filtered map (ai_id -> ai_record).
    """
    if not filters:
        return candidate_map

    def keep(record):
        for k, v in filters.items():
            if k not in record:
                return False
            # simple equality or substring match
            val = record.get(k)
            if isinstance(v, str):
                if v.lower() not in str(val).lower():
                    return False
            else:
                if val != v:
                    return False
        return True

    return {aid: rec for aid, rec in candidate_map.items() if keep(rec)}

