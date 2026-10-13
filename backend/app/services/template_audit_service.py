"""
Read-Only Biometric Template Audit Service
==========================================
Audits existing enrolled templates in IrisEmbedding (V1), IrisEmbeddingV2 (V2),
and FaceEmbedding tables for corrupt, blank, zeroed, non-finite, degenerate,
or orphaned records.

STRICT SAFETY RULE:
This service is strictly READ-ONLY. It NEVER issues INSERT, UPDATE, or DELETE
statements, ensuring existing database records remain unaltered.
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np

# Ensure backend directory is in sys.path for standalone script execution
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbedding, IrisEmbeddingV2, FaceEmbedding


def audit_enrolled_templates(verbose: bool = True) -> Dict[str, Any]:
    """
    Scans all registered biometric templates in the database and flags any
    suspicious records without modifying or deleting anything.

    Returns:
        Dict containing total counts, clean counts, suspicious counts,
        and details of any flagged records.
    """
    db = SessionLocal()
    report: Dict[str, Any] = {
        "status": "success",
        "summary": {
            "total_users": 0,
            "iris_v1": {"total": 0, "clean": 0, "suspicious": 0},
            "iris_v2": {"total": 0, "clean": 0, "suspicious": 0},
            "face": {"total": 0, "clean": 0, "suspicious": 0},
            "total_templates": 0,
            "total_suspicious": 0
        },
        "suspicious_records": [],
        "duplicate_groups": []
    }

    try:
        # Load all valid user IDs for foreign key reference checks
        users = db.query(User).all()
        user_ids = {u.id for u in users}
        user_map = {u.id: u.name for u in users}
        report["summary"]["total_users"] = len(users)

        # 1. Audit V1 Iris Templates (576-D)
        v1_records = db.query(IrisEmbedding).all()
        report["summary"]["iris_v1"]["total"] = len(v1_records)
        for rec in v1_records:
            issues = _check_embedding(
                rec.embedding,
                expected_dim=576,
                check_norm=True,
                expected_norm=1.0,
                norm_tolerance=0.08
            )
            if rec.user_id not in user_ids:
                issues.append(f"ORPHANED_RECORD: user_id #{rec.user_id} does not exist in users table")

            if issues:
                report["summary"]["iris_v1"]["suspicious"] += 1
                report["suspicious_records"].append({
                    "table": "iris_embeddings (v1)",
                    "id": rec.id,
                    "user_id": rec.user_id,
                    "user_name": user_map.get(rec.user_id, "UNKNOWN"),
                    "issues": issues
                })
            else:
                report["summary"]["iris_v1"]["clean"] += 1

        # 2. Audit V2 Iris Templates (512-D)
        v2_records = db.query(IrisEmbeddingV2).all()
        report["summary"]["iris_v2"]["total"] = len(v2_records)
        for rec in v2_records:
            issues = _check_embedding(
                rec.embedding,
                expected_dim=512,
                check_norm=True,
                expected_norm=1.0,
                norm_tolerance=0.08
            )
            if rec.user_id not in user_ids:
                issues.append(f"ORPHANED_RECORD: user_id #{rec.user_id} does not exist in users table")
            if rec.eye_side not in ("L", "R"):
                issues.append(f"INVALID_EYE_SIDE: '{rec.eye_side}' (expected 'L' or 'R')")

            if issues:
                report["summary"]["iris_v2"]["suspicious"] += 1
                report["suspicious_records"].append({
                    "table": "iris_embeddings_v2",
                    "id": rec.id,
                    "user_id": rec.user_id,
                    "user_name": user_map.get(rec.user_id, "UNKNOWN"),
                    "eye_side": rec.eye_side,
                    "issues": issues
                })
            else:
                report["summary"]["iris_v2"]["clean"] += 1

        # 3. Audit Face Templates (128-D)
        face_records = db.query(FaceEmbedding).all()
        report["summary"]["face"]["total"] = len(face_records)
        for rec in face_records:
            issues = _check_embedding(
                rec.embedding,
                expected_dim=128,
                check_norm=True,
                expected_norm=1.0,
                norm_tolerance=0.05
            )
            if rec.user_id not in user_ids:
                issues.append(f"ORPHANED_RECORD: user_id #{rec.user_id} does not exist in users table")

            if issues:
                report["summary"]["face"]["suspicious"] += 1
                report["suspicious_records"].append({
                    "table": "face_embeddings",
                    "id": rec.id,
                    "user_id": rec.user_id,
                    "user_name": user_map.get(rec.user_id, "UNKNOWN"),
                    "issues": issues
                })
            else:
                report["summary"]["face"]["clean"] += 1

        # Check for identical duplicate embedding vectors within V2
        v2_hashes: Dict[str, List[int]] = {}
        for rec in v2_records:
            if rec.embedding and isinstance(rec.embedding, list):
                arr = np.array(rec.embedding, dtype=np.float32)
                h = arr.tobytes().hex()[:32]
                v2_hashes.setdefault(h, []).append(rec.id)
        for h, ids in v2_hashes.items():
            if len(ids) > 1:
                report["duplicate_groups"].append({"table": "iris_embeddings_v2", "matching_ids": ids})

        # Calculate totals
        total_templates = (
            report["summary"]["iris_v1"]["total"]
            + report["summary"]["iris_v2"]["total"]
            + report["summary"]["face"]["total"]
        )
        total_suspicious = len(report["suspicious_records"])
        report["summary"]["total_templates"] = total_templates
        report["summary"]["total_suspicious"] = total_suspicious

        if verbose:
            _print_summary(report)

        return report

    finally:
        # Strictly read-only: close session without altering records
        db.close()


def _check_embedding(
    emb_data: Any,
    expected_dim: int,
    check_norm: bool = True,
    expected_norm: float = 1.0,
    norm_tolerance: float = 0.05
) -> List[str]:
    """Helper to audit single embedding payload against integrity criteria."""
    issues = []
    if emb_data is None:
        issues.append("NULL_EMBEDDING")
        return issues

    if not isinstance(emb_data, (list, np.ndarray)):
        issues.append(f"INVALID_TYPE: expected list, got {type(emb_data).__name__}")
        return issues

    dim = len(emb_data)
    if dim != expected_dim:
        issues.append(f"DIMENSION_MISMATCH: expected {expected_dim}, got {dim}")

    if dim == 0:
        issues.append("EMPTY_EMBEDDING_VECTOR")
        return issues

    try:
        arr = np.array(emb_data, dtype=np.float32)
    except Exception as e:
        issues.append(f"UNPARSEABLE_ARRAY: {e}")
        return issues

    if not np.all(np.isfinite(arr)):
        issues.append("NON_FINITE_VALUES: contains NaN, Inf, or -Inf")

    if np.all(arr == 0):
        issues.append("ALL_ZERO_VECTOR")

    norm = float(np.linalg.norm(arr))
    if norm < 1e-4:
        issues.append(f"NEAR_ZERO_NORM: {norm:.6f}")
    elif check_norm:
        if abs(norm - expected_norm) > norm_tolerance:
            issues.append(f"ABNORMAL_L2_NORM: expected ~{expected_norm:.2f}, got {norm:.4f}")

    variance = float(np.var(arr))
    if variance < 1e-7:
        issues.append(f"NEAR_ZERO_VARIANCE: degenerate constant embedding (var={variance:.2e})")

    return issues


def _print_summary(report: Dict[str, Any]):
    print("=" * 60)
    print(" [AUDIT] READ-ONLY BIOMETRIC TEMPLATE AUDIT REPORT")
    print("=" * 60)
    s = report["summary"]
    print(f" Total Registered Users: {s['total_users']}")
    print(f" Iris V1 Templates     : {s['iris_v1']['total']} (Clean: {s['iris_v1']['clean']}, Suspicious: {s['iris_v1']['suspicious']})")
    print(f" Iris V2 Templates     : {s['iris_v2']['total']} (Clean: {s['iris_v2']['clean']}, Suspicious: {s['iris_v2']['suspicious']})")
    print(f" Face Templates        : {s['face']['total']} (Clean: {s['face']['clean']}, Suspicious: {s['face']['suspicious']})")
    print(f" Total Templates       : {s['total_templates']}")
    print(f" Total Suspicious      : {s['total_suspicious']}")

    if report["suspicious_records"]:
        print("\n [!] Suspicious Records Found:")
        for item in report["suspicious_records"]:
            print(f"   - Table: {item['table']}, ID: {item['id']}, User: {item['user_name']} (#{item['user_id']})")
            for iss in item["issues"]:
                print(f"       * {iss}")
    else:
        print("\n [OK] All enrolled database templates passed integrity checks.")

    if report["duplicate_groups"]:
        print(f"\n [!] Exact Duplicate Template Groups: {len(report['duplicate_groups'])}")
    print("=" * 60)


if __name__ == "__main__":
    audit_enrolled_templates(verbose=True)
