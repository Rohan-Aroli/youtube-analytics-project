import { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import ChartWrapper from '../components/charts/ChartWrapper';

export default function CreatorComparison() {
  const [ids, setIds] = useState("1,2");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCompare = () => {
    const idArray = ids.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    if (idArray.length < 2 || idArray.length > 4) {
      setError("Please provide between 2 and 4 valid Creator IDs.");
      return;
    }
    setError("");
    setLoading(true);

    const params = new URLSearchParams();
    idArray.forEach(id => params.append("creator_ids", id.toString()));

    axios.get(`/api/creators/compare?${params.toString()}`)
      .then(res => {
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error fetching comparison");
        setLoading(false);
      });
  };

  const getChartData = (metric: string) => {
    if (!data) return [];
    return Object.keys(data.comparison_metrics[metric]).map(youtuber => ({
      name: youtuber,
      value: data.comparison_metrics[metric][youtuber]
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Creator Comparison</h1>
        <p className="text-slate-500">Compare metrics between 2 to 4 creators.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <label className="block text-sm font-medium text-slate-700 mb-2">Enter Creator IDs (comma separated, e.g. 1, 2, 3)</label>
        <div className="flex gap-4">
          <input
            type="text"
            value={ids}
            onChange={e => setIds(e.target.value)}
            className="flex-1 rounded-md border-slate-300 shadow-sm border p-2 focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="1, 2, 3"
          />
          <button
            onClick={handleCompare}
            className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700"
          >
            Compare
          </button>
        </div>
        {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
      </div>

      {loading && <div>Loading comparison...</div>}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartWrapper title="Subscribers Comparison">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getChartData('subscribers')}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="value" fill="#3b82f6" name="Subscribers" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper title="Video Views Comparison">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getChartData('video_views')}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="value" fill="#8b5cf6" name="Views" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper title="Highest Yearly Earnings ($)">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getChartData('highest_yearly_earnings')}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="value" fill="#10b981" name="Earnings" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>

          <ChartWrapper title="Total Uploads">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getChartData('uploads')}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="value" fill="#f59e0b" name="Uploads" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </div>
      )}
    </div>
  );
}
