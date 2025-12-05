"""
API Routes for AI Recommendation System.

Endpoints:
- POST /api/recommend     → personalized + semantic hybrid recommendations
- POST /api/deep_search   → deep LLM-enhanced ranking + personalization
- POST /api/feedback      → store like/dislike feedback
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Core services
from backend.services.recommender import recommend_ai, deep_search_ai, normalize_records

# Personalized hybrid logic
from backend.services.hybrid_recommend import (
    get_user_personal_scores,
    get_collaborative_scores,
)

# Database utilities
from backend.utils.db import save_feedback, get_all_ai, log_user_action,get_bookmarks

# -------------------------
# ROUTER PREFIX
# -------------------------
router = APIRouter()


# -----------------------------------------
# Request Schemas
# -----------------------------------------
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    user_id: int
    ai_id: int
    feedback: Optional[int] = None      # 1 or -1
    rating: Optional[int] = None        # 1–5
    comment: Optional[str] = None
    bookmark: Optional[bool] = None     # True = saved


class LogEventRequest(BaseModel):
    user_id: Optional[int] = 0
    ai_id: Optional[int] = None
    action: str
    query: Optional[str] = None
    dwell_time: Optional[int] = None


def apply_filters(items, filters):
    if not filters:
        return items

    cost = filters.get("cost")
    lang = filters.get("languages")
    cat = filters.get("category")

    filtered = []
    for item in items:
        ok = True

        if cost and item.get("cost") != cost:
            ok = False

        if lang and lang not in (item.get("languages") or ""):
            ok = False

        if cat and item.get("category") != cat:
            ok = False

        if ok:
            filtered.append(item)

    return filtered

# -----------------------------------------
# Utility Normalization
# -----------------------------------------
def format_for_frontend(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"results": normalize_records(records)}


# -----------------------------------------
# INTERNAL HELPER — MAP CSV ai_ids → DB ai_ids
# -----------------------------------------
def build_ai_map(ai_data):
    """
    Create a dict mapping:
        lowercased name → real DB ai_id
    """
    return {row["name"].strip().lower(): row["ai_id"] for row in ai_data}


def fix_ai_id(item, name_to_id):
    name = (item.get("name") or item.get("Name") or "").strip().lower()

    # If name matches → assign correct ai_id
    if name in name_to_id:
        item["ai_id"] = name_to_id[name]
        return item

    # If name does not match → skip this item completely
    return None




# -----------------------------------------
# QUICK RECOMMENDATION (Personalized Hybrid)
# -----------------------------------------
@router.post("/recommend")
def recommend_route(req: RecommendRequest):
    try:
        user_id = req.user_id or 0

        # Step 1: Semantic initial candidates
        base_results = recommend_ai(req.query, req.top_k)
        if not base_results:
            return {"results": []}

        # Step 2: Fetch AI dataset
        ai_data = get_all_ai() or []
        if not ai_data:
            return {"results": base_results}

        # Build mapping
        ai_map = build_ai_map(ai_data)

        # Apply mapping to every item
        base_results = [fix_ai_id(item, ai_map) for item in base_results]

        # Step 3: Personalized scoring
        personal_scores = get_user_personal_scores(user_id, req.top_k)

        # Step 4: Collaborative scoring
        collab_scores = get_collaborative_scores(user_id, ai_data)

        # Step 5: Combine scores
        base_results = [i for i in base_results if i]
        for idx, item in enumerate(base_results):
            ai_id = item.get("ai_id")
            p_score = personal_scores.get(ai_id, 0.0)
            c_score = collab_scores.get(ai_id, 0.0)

            if user_id:
                final_score = (0.7 * p_score) + (0.3 * c_score)
            else:
                final_score = c_score

            item["score"] = final_score

        # Step 6: Sort results


        base_results = [i for i in base_results if i]
        base_results = apply_filters(base_results, req.filters or {})



        # Step 7: Log implicit event
        for item in base_results:
            log_user_action(
                user_id=user_id,
                ai_id=item["ai_id"],
                query=req.query,
                action="search",
                feedback=None
            )

        return {"results": base_results[:req.top_k]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommend route failed: {e}")


# -----------------------------------------
# DEEP SEARCH ROUTE (LLM + Personalization)
# -----------------------------------------
@router.post("/deep_search")
def deep_search_route(req: RecommendRequest):
    try:
        user_id = req.user_id or 0

        # Step 1: Deep LLM candidate extraction
        deep_results = deep_search_ai(req.query, req.top_k)
        if not deep_results:
            return {"results": []}

        # Step 2: AI dataset
        ai_data = get_all_ai() or []
        if not ai_data:
            return {"results": deep_results}

        # Build mapping
        ai_map = build_ai_map(ai_data)

        # Apply mapping
        deep_results = [fix_ai_id(item, ai_map) for item in deep_results]

        # Step 3: Personalized score
        personal_scores = get_user_personal_scores(user_id, req.top_k)

        # Step 4: Collaborative score
        collab_scores = get_collaborative_scores(user_id, ai_data)

        # Step 5: Final hybrid score
        for idx, item in enumerate(deep_results):
            ai_id = item.get("ai_id")
            p_score = personal_scores.get(ai_id, 0.0)
            c_score = collab_scores.get(ai_id, 0.0)

            if user_id:
                final_score = (0.7 * p_score) + (0.3 * c_score)
            else:
                final_score = c_score

            item["score"] = final_score


        deep_results = [fix_ai_id(item, ai_map) for item in deep_results]
        deep_results = [i for i in deep_results if i]  # FIX


        # Apply filters
        deep_results = apply_filters(deep_results, req.filters)



        # Step 6: Log implicit deep search action
        for item in deep_results:
            log_user_action(
                user_id=user_id,
                ai_id=item["ai_id"],
                query=req.query,
                action="deep_search",
                feedback=None
            )

        return {"results": deep_results[:req.top_k]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep search route failed: {e}")


# -----------------------------------------
# FEEDBACK ROUTE (EXTENDED)
# -----------------------------------------
@router.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    try:
        if not req.user_id or not req.ai_id:
            raise HTTPException(status_code=400, detail="Missing user_id or ai_id")

        # Store extended feedback
        bm = None
        if req.bookmark is True: bm = 1
        if req.bookmark is False: bm = 0

        save_feedback(
            user_id=req.user_id,
            ai_id=req.ai_id,
            feedback=req.feedback,
            rating=req.rating,
            comment=req.comment,
            sentiment=None,
            bookmark=bm
        )


        # Log implicit
        log_user_action(
            user_id=req.user_id,
            ai_id=req.ai_id,
            query="",
            action="rate" if req.rating else ("bookmark" if req.bookmark is not None else "feedback")
,
            feedback=req.feedback
        )

        return {"status": "success", "message": "Feedback saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {e}")

# ============================================================
# LOG EVENT ROUTE (implicit feedback)
# ============================================================
@router.post("/log_event")
def log_route(req: LogEventRequest):
    try:
        log_user_action(
            user_id=req.user_id or 0,
            ai_id=req.ai_id,
            query=req.query or "",
            action=req.action,
            dwell_time=req.dwell_time
        )
        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(500, f"Log event failed: {e}")

# ============================================================
# BOOKMARK ROUTE
# ============================================================
@router.get("/bookmarks")
def bookmarks_route(user_id: int):
    if not user_id:
           return {"bookmarks": []}
    return {"bookmarks": get_bookmarks(user_id)}

 

# -----------------------------------------
# HEALTH CHECK
# -----------------------------------------
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Recommender API"}


