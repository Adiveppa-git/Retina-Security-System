-- Iris Security System MySQL Schema
CREATE DATABASE IF NOT EXISTS iris_security_db;
USE iris_security_db;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Iris Feature Embeddings Table (512-Dimensional Vector stored as JSON)
CREATE TABLE IF NOT EXISTS iris_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    embedding JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. Access Logs Table
CREATE TABLE IF NOT EXISTS access_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL, -- ACCESS_GRANTED, ACCESS_DENIED, SPOOF_DETECTED
    action VARCHAR(50) DEFAULT 'CHECK-IN', -- CHECK-IN or CHECK-OUT
    similarity_score FLOAT DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Face Feature Embeddings Table (128-Dimensional Vector stored as JSON)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    embedding JSON NOT NULL, -- 128-D float vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 5. Iris Feature Embeddings V2 Table (512-Dimensional Vector stored as JSON)
CREATE TABLE IF NOT EXISTS iris_embeddings_v2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    eye_side VARCHAR(10) NOT NULL DEFAULT 'L',
    model_version VARCHAR(20) NOT NULL DEFAULT 'v2',
    embedding_dimension INT NOT NULL DEFAULT 512,
    embedding JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
