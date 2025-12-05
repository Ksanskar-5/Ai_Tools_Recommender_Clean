from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.db import register_user, verify_user

router = APIRouter()

class Register(BaseModel):
    name: str
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

# /api/auth/signup   (after prefix)
@router.post("/signup")
def register_user_route(user: Register):
    user_id = register_user(user.name, user.email, user.password)
    if not user_id:
        raise HTTPException(status_code=400, detail="User already exists or registration failed")
    return {
        "user_id": user_id,
        "email": user.email,
        "message": "Registration successful"
    }

# /api/auth/login   (after prefix)
@router.post("/login")
def login_user_route(user: Login):
    user_data = verify_user(user.email, user.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "user_id": user_data["user_id"],
        "email": user.email,
        "message": "Login successful"
    }
