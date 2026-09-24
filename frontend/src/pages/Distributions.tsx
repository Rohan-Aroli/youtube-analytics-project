import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import ChartWrapper from '../components/charts/ChartWrapper';
import StatCard from '../components/statistics/StatCard';

export default function Distributions() {
  const [variable, setVariable] = useState("subscribers");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const variables = [
    { value: "subscribers", label: "Subscribers" },
    { value: "video_views", label: "Video Views" },
    { value: "uploads", label: "Uploads" },
    { value: "highest_yearly_earnings", label: "Highest Yearly Earnings" }
  ];

  const fetchDistribution = () => {
    setLoading(true);
    setError("");
    axios.get(`/api/statistics/distribution?variable=${variable}`)
      .then(res => {
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error fetching distribution");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchDistribution();
  }, [variable]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Distribution Analysis</h1>
        <p className="text-slate-500">Analyze the normality and shape of numeric distributions.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <label className="block text-sm font-medium text-slate-700 mb-2">Select Variable</label>
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

      {loading && <div>Loading distribution data...</div>}
      {error && <div className="text-red-500">{error}</div>}

      {data && !loading && (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-100 p-5 rounded-lg text-blue-900">
            <h3 className="font-semibold mb-2">Normality Assessment & Interpretation</h3>
            <p>{data.interpretation}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <StatCard title="Shapiro-Wilk Statistic" value={data.shapiro_wilk_stat?.toFixed(4) || "N/A"} />
            <StatCard
              title="Shapiro-Wilk p-value"
              value={data.shapiro_wilk_p?.toExponential(4) || "N/A"}
              subtitle={data.is_normal ? "Distribution appears Normal" : "Distribution is NOT Normal"}
            />
          </div>

          <ChartWrapper title="Histogram (Frequency Distribution)" description="Shows the count of channels within automatic bin ranges.">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.histogram_data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tickFormatter={(val) => new Intl.NumberFormat('en-US', {notation: 'compact'}).format(val)} />
                <YAxis />
                <RechartsTooltip formatter={(value) => [value, "Count"]} labelFormatter={(label) => `Range starting: ${label}`} />
                <Bar dataKey="count" fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        </div>
      )}
    </div>
  );
}
