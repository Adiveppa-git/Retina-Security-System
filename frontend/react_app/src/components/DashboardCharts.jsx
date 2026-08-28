import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";

import { Pie, Line } from "react-chartjs-2";

ChartJS.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function DashboardCharts({ data }) {

  const pieData = {
    labels: ["Authorized", "Denied"],
    datasets: [
      {
        label: "Verification",
        data: [
          data.authorized,
          data.denied
        ],
        backgroundColor: [
          "#22c55e",
          "#ef4444"
        ],
        borderColor: [
          "#16a34a",
          "#dc2626"
        ],
        borderWidth: 2
      }
    ]
  };

  const pieOptions = {

    responsive: true,

    plugins: {

      legend: {

        position: "bottom"

      },

      title: {

        display: true,

        text: "Verification Statistics"

      }

    }

  };

  const lineData = {

    labels: [
      "Mon",
      "Tue",
      "Wed",
      "Thu",
      "Fri",
      "Sat",
      "Sun"
    ],

    datasets: [

      {

        label: "Verification Requests",

        data: [
          2,
          5,
          8,
          4,
          7,
          6,
          data.total_requests
        ],

        borderColor: "#2563eb",

        backgroundColor: "rgba(37,99,235,0.20)",

        pointBackgroundColor: "#2563eb",

        pointRadius: 5,

        fill: true,

        tension: 0.4

      }

    ]

  };

  const lineOptions = {

    responsive: true,

    plugins: {

      legend: {

        display: true

      },

      title: {

        display: true,

        text: "Weekly Requests"

      }

    },

    scales: {

      y: {

        beginAtZero: true

      }

    }

  };

  return (

    <div className="charts-grid">

      <div className="chart-card">

        <Pie

          data={pieData}

          options={pieOptions}

        />

      </div>

      <div className="chart-card">

        <Line

          data={lineData}

          options={lineOptions}

        />

      </div>

    </div>

  );

}

export default DashboardCharts;