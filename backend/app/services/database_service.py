from app.database.connection import SessionLocal
from app.models.user_model import User

def add_user(name, iris_path):
    db = SessionLocal()

    user = User(name=name, iris_path=iris_path)

    db.add(user)
    db.commit()
    db.close()


def get_user_by_id(user_id):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    return user