import { useState, useEffect } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';
import DataTable from '../components/tables/DataTable';

export default function Correlation() {
  const [var1, setVar1] = useState("subscribers");
  const [var2, setVar2] = useState("video_views");
  const [method, setMethod] = useState("pearson");

  const [data, setData] = useState<any>(null);
  const [matrixData, setMatrixData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const variables = [
    { value: "subscribers", label: "Subscribers" },
    { value: "video_views", label: "Video Views" },
    { value: "uploads", label: "Uploads" },
    { value: "highest_yearly_earnings", label: "Highest Yearly Earnings" },
    { value: "population", label: "Population" },
    { value: "unemployment_rate", label: "Unemployment Rate" }
  ];

  const fetchCorrelation = () => {
    setLoading(true);
    setError("");
    axios.get(`/api/statistics/correlation?var1=${var1}&var2=${var2}&method=${method}`)
      .then(res => {
        setData(res.data);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error fetching correlation");
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const fetchMatrix = () => {
    axios.get(`/api/statistics/correlation/matrix?method=${method}`)
      .then(res => {
        setMatrixData(res.data);
      })
      .catch(err => console.error("Matrix error", err));
  };

  useEffect(() => {
    fetchCorrelation();
    fetchMatrix();
  }, [var1, var2, method]);

  const renderMatrix = () => {
    if (!matrixData) return null;
    const cols = [{ header: "Variables", accessor: "variable" }, ...matrixData.variables.map((v: string) => ({ header: v, accessor: v }))];
    const tableData = matrixData.variables.map((v: string, i: number) => {
      const row: any = { variable: v };
      matrixData.matrix[i].forEach((val: number, j: number) => {
        row[matrixData.variables[j]] = val !== null ? val.toFixed(4) : "N/A";
      });
      return row;
    });

    return <DataTable columns={cols} data={tableData} />;
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Correlation Analysis</h1>
        <p className="text-slate-500">Analyze relationships between variables.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Variable 1</label>
            <select value={var1} onChange={(e) => setVar1(e.target.value)} className="w-full rounded-md border-slate-300 shadow-sm border p-2 bg-white">
              {variables.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Variable 2</label>
            <select value={var2} onChange={(e) => setVar2(e.target.value)} className="w-full rounded-md border-slate-300 shadow-sm border p-2 bg-white">
              {variables.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)} className="w-full rounded-md border-slate-300 shadow-sm border p-2 bg-white">
              <option value="pearson">Pearson</option>
              <option value="spearman">Spearman</option>
            </select>
          </div>
        </div>
      </div>

      {loading && <div>Calculating correlation...</div>}
      {error && <div className="text-red-500">{error}</div>}

      {data && !loading && (
        <div className="space-y-6">
          <div className="bg-emerald-50 border border-emerald-100 p-5 rounded-lg text-emerald-900">
            <h3 className="font-semibold mb-2">Statistical Interpretation</h3>
            <p>{data.interpretation}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard title="Correlation Coefficient (r)" value={data.coefficient.toFixed(4)} subtitle={`Method: ${data.method}`} />
            <StatCard title="p-value" value={data.p_value.toExponential(4)} subtitle={data.p_value < 0.05 ? "Statistically Significant" : "Not Significant"} />
            <StatCard title="Sample Size (n)" value={data.sample_size} />
          </div>

          <div className="mt-8">
            <h2 className="text-xl font-semibold text-slate-800 mb-4">Correlation Matrix</h2>
            {renderMatrix()}
          </div>
        </div>
      )}
    </div>
  );
}
