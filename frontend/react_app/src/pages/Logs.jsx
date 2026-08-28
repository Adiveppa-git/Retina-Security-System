import { useEffect, useState } from "react";
import api from "../services/api";
import "../styles/Logs.css";

function Logs() {

    const [logs, setLogs] = useState([]);

    useEffect(() => {

        fetchLogs();

    }, []);

    const fetchLogs = async () => {

        try {

            const response = await api.get("/logs");

            setLogs(response.data);

        }

        catch (error) {

            console.log(error);

        }

    };

    return (

        <div className="logs-container">

            <h1>Access Logs</h1>

            <table>

                <thead>

                    <tr>

                        <th>Date</th>
                        <th>Time</th>
                        <th>Person</th>
                        <th>Similarity</th>
                        <th>Status</th>
                        <th>Response Time</th>

                    </tr>

                </thead>

                <tbody>

                    {logs.map((log, index) => (

                        <tr key={index}>

                            <td>{log.Date}</td>

                            <td>{log.Time}</td>

                            <td>{log.Person}</td>

                            <td>{log.Similarity}</td>

                            <td>

                                <span
                                    className={
                                        log.Status === "Success"
                                            ? "success"
                                            : log.Status === "Fake Iris"
                                            ? "fake"
                                            : "denied"
                                    }
                                >

                                    {log.Status}

                                </span>

                            </td>

                            <td>{log["Response Time"]}</td>

                        </tr>

                    ))}

                </tbody>

            </table>

        </div>

    );

}

export default Logs;