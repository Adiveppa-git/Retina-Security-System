import os
import torch

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

USER_DB = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/user_features.pth"
)


def get_all_users():

    if not os.path.exists(USER_DB):
        return []

    try:
        db = torch.load(USER_DB)

        users = []

        for index, name in enumerate(db["names"], start=1):

            users.append({
                "id": index,
                "name": name
            })

        return users

    except Exception as e:

        print(e)

        return []