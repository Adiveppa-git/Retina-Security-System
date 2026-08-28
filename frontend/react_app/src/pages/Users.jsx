import { useEffect, useState } from "react";
import api from "../services/api";
import "../styles/Users.css";

import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button
} from "@mui/material";

function Users() {

  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");

  const [open, setOpen] = useState(false);
  const [oldName, setOldName] = useState("");
  const [newName, setNewName] = useState("");

  // ------------------------
  // Load Users
  // ------------------------
  const loadUsers = async () => {

    try {

      const response = await api.get("/user/all");

      setUsers(response.data);

    } catch (err) {

      console.log(err);

    }

  };

  useEffect(() => {

    loadUsers();

  }, []);

  // ------------------------
  // Delete User
  // ------------------------
  const deleteUser = async (name) => {

    if (!window.confirm(`Delete "${name}" ?`))
      return;

    try {

      const response = await api.delete(`/user/${name}`);

      alert(response.data.message);

      loadUsers();

    } catch {

      alert("Delete Failed");

    }

  };

  // ------------------------
  // Open Rename Dialog
  // ------------------------
  const openDialog = (name) => {

    setOldName(name);

    setNewName(name);

    setOpen(true);

  };

  // ------------------------
  // Rename User
  // ------------------------
  const renameUser = async () => {

    try {

      const formData = new FormData();

      formData.append("old_name", oldName);

      formData.append("new_name", newName);

      const response = await api.put(
        "/user/update",
        formData
      );

      alert(response.data.message);

      setOpen(false);

      loadUsers();

    } catch (err) {

      console.log(err);

      alert("Update Failed");

    }

  };

  const filteredUsers = users.filter((u) =>
    u.name.toLowerCase().includes(search.toLowerCase())
  );

  return (

    <div className="users-container">

      <h1>Registered Users</h1>

      <input
        className="search-box"
        placeholder="Search User..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <h2>Total Users : {filteredUsers.length}</h2>

      <table className="users-table">

        <thead>

          <tr>

            <th>S.No</th>

            <th>User ID</th>

            <th>Rename</th>

            <th>Delete</th>

          </tr>

        </thead>

        <tbody>

          {filteredUsers.map((user) => (

            <tr key={user.id}>

              <td>{user.id}</td>

              <td>{user.name}</td>

              <td>

                <button
                  className="update-btn"
                  onClick={() => openDialog(user.name)}
                >
                  Rename
                </button>

              </td>

              <td>

                <button
                  className="delete-btn"
                  onClick={() => deleteUser(user.name)}
                >
                  Delete
                </button>

              </td>

            </tr>

          ))}

        </tbody>

      </table>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
      >

        <DialogTitle>

          Rename User

        </DialogTitle>

        <DialogContent>

          <TextField
            fullWidth
            margin="dense"
            label="New User ID"
            value={newName}
            onChange={(e)=>setNewName(e.target.value)}
          />

        </DialogContent>

        <DialogActions>

          <Button
            onClick={()=>setOpen(false)}
          >
            Cancel
          </Button>

          <Button
            variant="contained"
            onClick={renameUser}
          >
            Update
          </Button>

        </DialogActions>

      </Dialog>

    </div>

  );

}

export default Users;