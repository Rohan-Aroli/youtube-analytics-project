import { useEffect, useState } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';
import { Users, Eye, DollarSign, ListFilter, Globe } from 'lucide-react';

interface DashboardSummary {
  total_creators: number;
  mean_subscribers: number;
  median_subscribers: number;
  mean_views: number;
  median_views: number;
  mean_earnings: number;
  median_earnings: number;
  num_categories: number;
  num_countries: number;
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/dashboard/summary')
      .then(res => {
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching dashboard summary:", err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading...</div>;
  if (!data) return <div>Error loading data.</div>;

  const formatNumber = (num: number) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(num);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Platform Overview</h1>
        <p className="text-slate-500">Summary statistics for the YouTube Creator Analytics platform.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Creators"
          value={data.total_creators.toLocaleString()}
          icon={<Users />}
        />
        <StatCard
          title="Categories"
          value={data.num_categories}
          icon={<ListFilter />}
        />
        <StatCard
          title="Countries"
          value={data.num_countries}
          icon={<Globe />}
        />
      </div>

      <h2 className="text-xl font-semibold text-slate-800 mt-8 mb-4">Key Metrics</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard
          title="Subscribers"
          value={`Mean: ${formatNumber(data.mean_subscribers)}`}
          subtitle={`Median: ${formatNumber(data.median_subscribers)}`}
          icon={<Users className="text-blue-500"/>}
        />
        <StatCard
          title="Video Views"
          value={`Mean: ${formatNumber(data.mean_views)}`}
          subtitle={`Median: ${formatNumber(data.median_views)}`}
          icon={<Eye className="text-indigo-500"/>}
        />
        <StatCard
          title="Highest Yearly Earnings"
          value={`Mean: $${formatNumber(data.mean_earnings)}`}
          subtitle={`Median: $${formatNumber(data.median_earnings)}`}
          icon={<DollarSign className="text-green-500"/>}
        />
      </div>
    </div>
  );
}
