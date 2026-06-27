from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from app.core.auth import create_access_token, get_current_user
from app.core.user_service import get_user_by_email, create_user, verify_password
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    name:     str
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    name:         str
    email:        str


@router.post("/register", status_code=201)
def register(body: RegisterRequest):
    """Register a new user with name, email and password."""
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    create_user(name=body.name, email=body.email, password=body.password)
    logger.info(f"New user registered: {body.email}")
    return {
        "message": "Account created successfully. Please login to get your token.",
        "email":   body.email,
        "name":    body.name,
    }


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Login with email + password.
    Note: enter your email in the 'username' field.
    """
    user = get_user_by_email(form.username)
    if not user or not verify_password(form.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    token = create_access_token(data={"sub": user["email"]})
    logger.info(f"User logged in: {user['email']}")
    return TokenResponse(access_token=token, token_type="bearer", name=user["name"], email=user["email"])


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Returns current logged-in user's profile."""
    return user