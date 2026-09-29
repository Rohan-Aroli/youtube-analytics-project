import { useState } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';
import DataTable from '../components/tables/DataTable';

export default function Regression() {
  const [depVar, setDepVar] = useState("highest_yearly_earnings");
  const [indepVars, setIndepVars] = useState("subscribers, video_views, uploads");

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runRegression = () => {
    const varsArray = indepVars.split(",").map(s => s.trim()).filter(s => s !== "");
    if (varsArray.length === 0) {
      setError("Please provide at least one independent variable.");
      return;
    }

    setLoading(true);
    setError("");
    setData(null);

    axios.post(`/api/statistics/regression`, {
      dependent_variable: depVar,
      independent_variables: varsArray
    })
      .then(res => {
        setData(res.data);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error running regression");
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const getCoefColumns = () => [
    { header: "Variable", accessor: "variable" },
    { header: "Coefficient", accessor: "coefficient" },
    { header: "p-value", accessor: "p_value" },
    { header: "CI Lower", accessor: "ci_lower" },
    { header: "CI Upper", accessor: "ci_upper" },
    { header: "VIF", accessor: "vif" }
  ];

  const getCoefData = () => {
    if (!data) return [];
    return data.coefficients.map((c: any) => ({
      variable: c.variable,
      coefficient: c.coefficient.toExponential(4),
      p_value: c.p_value.toExponential(4),
      ci_lower: c.ci_lower.toExponential(4),
      ci_upper: c.ci_upper.toExponential(4),
      vif: data.diagnostics?.vif?.[c.variable] ? data.diagnostics.vif[c.variable].toFixed(2) : "N/A"
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Regression Analysis</h1>
        <p className="text-slate-500">Perform Simple and Multiple Linear Regression to model relationships.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dependent Variable (Y)</label>
            <input
              value={depVar}
              onChange={e => setDepVar(e.target.value)}
              className="w-full border rounded p-2 focus:border-indigo-500 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Independent Variables (X) - comma separated</label>
            <input
              value={indepVars}
              onChange={e => setIndepVars(e.target.value)}
              className="w-full border rounded p-2 focus:border-indigo-500 focus:ring-indigo-500"
            />
          </div>
        </div>
        <button onClick={runRegression} className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">Run Regression</button>
      </div>

      {loading && <div>Fitting model...</div>}
      {error && <div className="text-red-500">{error}</div>}

      {data && !loading && (
        <div className="space-y-6">
          <div className="bg-indigo-50 border border-indigo-100 p-5 rounded-lg text-indigo-900">
            <h3 className="font-semibold mb-2">Regression Equation</h3>
            <p className="font-mono text-sm overflow-x-auto">{data.equation}</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard title="R-Squared (R²)" value={data.r_squared.toFixed(4)} subtitle="Variance explained" />
            <StatCard title="Adjusted R²" value={data.adj_r_squared.toFixed(4)} />
            <StatCard title="F-Statistic p-value" value={data.f_pvalue.toExponential(4)} subtitle="Overall Model Significance" />
            <StatCard title="RMSE" value={data.rmse.toExponential(4)} />
          </div>

          <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Coefficients & Diagnostics</h3>
            <DataTable columns={getCoefColumns()} data={getCoefData()} />

            <div className="mt-4 text-sm text-slate-600">
              <p><strong>Note on Multicollinearity:</strong> VIF (Variance Inflation Factor) values &gt; 5 indicate potential multicollinearity, and &gt; 10 indicate strong multicollinearity concern.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
