import { useEffect, useState } from 'react';
import axios from 'axios';
import DataTable from '../components/tables/DataTable';

export default function DatasetExplorer() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const limit = 50;

  const fetchCreators = () => {
    setLoading(true);
    axios.get(`/api/creators?skip=${skip}&limit=${limit}`)
      .then(res => {
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching creators:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchCreators();
  }, [skip]);

  const columns = [
    { header: "Rank", accessor: "rank" },
    { header: "YouTuber", accessor: "youtuber" },
    { header: "Subscribers", accessor: "subscribers" },
    { header: "Video Views", accessor: "video_views" },
    { header: "Category", accessor: "category" },
    { header: "Country", accessor: "country" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Dataset Explorer</h1>
        <p className="text-slate-500">Explore the raw YouTube creator dataset.</p>
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
        {loading && <div className="mb-4">Loading data...</div>}
        <DataTable columns={columns} data={data} />

        <div className="flex justify-between items-center mt-4">
          <button
            disabled={skip === 0}
            onClick={() => setSkip(s => Math.max(0, s - limit))}
            className="px-4 py-2 bg-slate-100 text-slate-700 rounded-md disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-slate-600">Showing {skip + 1} - {skip + data.length}</span>
          <button
            disabled={data.length < limit}
            onClick={() => setSkip(s => s + limit)}
            className="px-4 py-2 bg-slate-100 text-slate-700 rounded-md disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
