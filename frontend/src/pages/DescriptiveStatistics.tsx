import { useState, useEffect } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';

export default function DescriptiveStatistics() {
  const [variable, setVariable] = useState("subscribers");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const variables = [
    { value: "subscribers", label: "Subscribers" },
    { value: "video_views", label: "Video Views" },
    { value: "uploads", label: "Uploads" },
    { value: "highest_yearly_earnings", label: "Highest Yearly Earnings" },
    { value: "lowest_yearly_earnings", label: "Lowest Yearly Earnings" },
    { value: "subscribers_for_last_30_days", label: "Subscribers Gained (Last 30 Days)" },
    { value: "video_views_for_the_last_30_days", label: "Views (Last 30 Days)" }
  ];

  const fetchStats = () => {
    setLoading(true);
    setError("");
    axios.get(`/api/statistics/descriptive?variable=${variable}`)
      .then(res => {
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error fetching statistics");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchStats();
  }, [variable]);

  const formatNumber = (num: number) => {
    if (num === null || num === undefined) return "N/A";
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(num);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Descriptive Statistics</h1>
        <p className="text-slate-500">Calculate core statistical measures for numerical variables.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <label className="block text-sm font-medium text-slate-700 mb-2">Select Variable to Analyze</label>
        <select
          value={variable}
          onChange={(e) => setVariable(e.target.value)}
          className="w-full md:w-1/3 rounded-md border-slate-300 shadow-sm border p-2 focus:border-indigo-500 focus:ring-indigo-500 bg-white"
        >
          {variables.map(v => (
            <option key={v.value} value={v.value}>{v.label}</option>
          ))}
        </select>
      </div>

      {loading && <div>Calculating statistics...</div>}
      {error && <div className="text-red-500">{error}</div>}

      {data && !loading && (
        <div className="space-y-6">
          <div className="bg-indigo-50 border border-indigo-100 p-5 rounded-lg text-indigo-900">
            <h3 className="font-semibold mb-2">Statistical Interpretation</h3>
            <p>{data.interpretation}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Count" value={formatNumber(data.count)} />
            <StatCard title="Mean" value={formatNumber(data.mean)} />
            <StatCard title="Median" value={formatNumber(data.median)} />
            <StatCard title="Mode" value={formatNumber(data.mode)} />

            <StatCard title="Minimum" value={formatNumber(data.min)} />
            <StatCard title="Maximum" value={formatNumber(data.max)} />
            <StatCard title="Range" value={formatNumber(data.range)} />
            <StatCard title="Standard Deviation" value={formatNumber(data.std_dev)} />

            <StatCard title="Q1 (25th Percentile)" value={formatNumber(data.q1)} />
            <StatCard title="Q3 (75th Percentile)" value={formatNumber(data.q3)} />
            <StatCard title="IQR" value={formatNumber(data.iqr)} />
            <StatCard title="Variance" value={formatNumber(data.variance)} />

            <StatCard title="Skewness" value={formatNumber(data.skewness)} subtitle={data.skewness > 1 || data.skewness < -1 ? "Highly Skewed" : "Acceptable"} />
            <StatCard title="Kurtosis" value={formatNumber(data.kurtosis)} subtitle={data.kurtosis > 3 ? "Leptokurtic (Heavy Tails)" : "Acceptable"} />
          </div>
        </div>
      )}
    </div>
  );
}
