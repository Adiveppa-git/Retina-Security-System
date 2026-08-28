from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

from backend.app.services.user_service import (
    register_user,
    get_all_users,
    delete_user,
    update_user
)

router = APIRouter(
    prefix="/user",
    tags=["User"]
)

# =====================================================
# Register User
# =====================================================

@router.post("/add")
async def add_user(
    name: str = Form(...),
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return register_user(name, image)


# =====================================================
# Get All Users
# =====================================================

@router.get("/all")
def all_users():

    return get_all_users()


# =====================================================
# Update User ID
# =====================================================

@router.put("/update")
def edit_user(

    old_name: str = Form(...),

    new_name: str = Form(...)

):

    return update_user(
        old_name,
        new_name
    )


# =====================================================
# Delete User
# =====================================================

@router.delete("/{user_name}")
def remove_user(user_name: str):

    return delete_user(user_name)