from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    embeddings = relationship("IrisEmbedding", back_populates="user", cascade="all, delete-orphan")
    embeddings_v2 = relationship("IrisEmbeddingV2", back_populates="user", cascade="all, delete-orphan")
    face_embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(JSON, nullable=False)  # 128-D float list
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="face_embeddings")


class IrisEmbedding(Base):
    __tablename__ = "iris_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(JSON, nullable=False)  # 512-D float list
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="embeddings")


class IrisEmbeddingV2(Base):
    __tablename__ = "iris_embeddings_v2"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    eye_side = Column(String(10), nullable=False, default="L")
    model_version = Column(String(20), nullable=False, default="v2")
    embedding_dimension = Column(Integer, nullable=False, default=512)
    embedding = Column(JSON, nullable=False)  # 512-D float list
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="embeddings_v2")


class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # ACCESS_GRANTED, ACCESS_DENIED, SPOOF_DETECTED
    action = Column(String(50), nullable=True, default="CHECK-IN")  # CHECK-IN or CHECK-OUT
    similarity_score = Column(Float, default=0.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


# Plural alias for convenience
AccessLogs = AccessLog


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), default="Administrator")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AdminSession(Base):
    __tablename__ = "admin_sessions"

    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

