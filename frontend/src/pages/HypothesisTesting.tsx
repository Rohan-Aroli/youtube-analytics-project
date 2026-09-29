import { useState } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';

export default function HypothesisTesting() {
  const [testType, setTestType] = useState("ttest");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<any>(null);

  // T-test state
  const [ttestVar, setTtestVar] = useState("highest_yearly_earnings");
  const [ttestGroup, setTtestGroup] = useState("category");
  const [ttestG1, setTtestG1] = useState("Music");
  const [ttestG2, setTtestG2] = useState("Entertainment");

  // ANOVA state
  const [anovaNum, setAnovaNum] = useState("highest_yearly_earnings");
  const [anovaCat, setAnovaCat] = useState("category");

  // Chi-Square state
  const [chi1, setChi1] = useState("category");
  const [chi2, setChi2] = useState("country");

  const runTest = () => {
    setLoading(true);
    setError("");
    setResult(null);

    let url = "";
    let payload = {};

    if (testType === "ttest") {
      url = "/api/statistics/hypothesis/t-test";
      payload = { variable: ttestVar, group_by: ttestGroup, group1_value: ttestG1, group2_value: ttestG2 };
    } else if (testType === "anova") {
      url = "/api/statistics/hypothesis/anova";
      payload = { numerical_variable: anovaNum, categorical_variable: anovaCat };
    } else if (testType === "chi") {
      url = "/api/statistics/hypothesis/chi-square";
      payload = { variable1: chi1, variable2: chi2 };
    }

    axios.post(`${url}`, payload)
      .then(res => {
        setResult(res.data);
      })
      .catch(err => {
        setError(err.response?.data?.detail || "Error running hypothesis test");
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Hypothesis Testing Lab</h1>
        <p className="text-slate-500">Run statistical tests to validate assumptions.</p>
      </div>

      <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">Select Test</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input type="radio" name="testType" value="ttest" checked={testType === "ttest"} onChange={() => setTestType("ttest")} />
              Independent T-Test
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" name="testType" value="anova" checked={testType === "anova"} onChange={() => setTestType("anova")} />
              One-way ANOVA
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" name="testType" value="chi" checked={testType === "chi"} onChange={() => setTestType("chi")} />
              Chi-Square
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {testType === "ttest" && (
            <>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Numerical Variable</label>
                <input value={ttestVar} onChange={e => setTtestVar(e.target.value)} className="w-full border rounded p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Grouping Variable (e.g., category)</label>
                <input value={ttestGroup} onChange={e => setTtestGroup(e.target.value)} className="w-full border rounded p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Group 1 Value</label>
                <input value={ttestG1} onChange={e => setTtestG1(e.target.value)} className="w-full border rounded p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Group 2 Value</label>
                <input value={ttestG2} onChange={e => setTtestG2(e.target.value)} className="w-full border rounded p-2" />
              </div>
            </>
          )}

          {testType === "anova" && (
            <>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Numerical Variable</label>
                <input value={anovaNum} onChange={e => setAnovaNum(e.target.value)} className="w-full border rounded p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Categorical Variable</label>
                <input value={anovaCat} onChange={e => setAnovaCat(e.target.value)} className="w-full border rounded p-2" />
              </div>
            </>
          )}

          {testType === "chi" && (
            <>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Categorical Variable 1</label>
                <input value={chi1} onChange={e => setChi1(e.target.value)} className="w-full border rounded p-2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Categorical Variable 2</label>
                <input value={chi2} onChange={e => setChi2(e.target.value)} className="w-full border rounded p-2" />
              </div>
            </>
          )}
        </div>

        <button onClick={runTest} className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">Run Test</button>
      </div>

      {loading && <div>Running test...</div>}
      {error && <div className="text-red-500">{error}</div>}

      {result && !loading && (
        <div className="space-y-6">
          <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4 border-b pb-2">{result.test_name}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm mb-6">
              <div><span className="font-semibold text-slate-700">Null Hypothesis (H0):</span> {result.null_hypothesis}</div>
              <div><span className="font-semibold text-slate-700">Alternative (H1):</span> {result.alternative_hypothesis}</div>
              <div><span className="font-semibold text-slate-700">Alpha Level:</span> {result.alpha}</div>
              <div><span className={`font-bold ${result.decision.includes('Reject') ? 'text-red-600' : 'text-slate-600'}`}>Decision:</span> {result.decision}</div>
            </div>

            <div className="bg-indigo-50 border border-indigo-100 p-4 rounded-lg text-indigo-900 mb-6">
              <span className="font-semibold block mb-1">Interpretation:</span>
              {result.interpretation}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard title="Test Statistic" value={result.statistic.toFixed(4)} />
              <StatCard title="p-value" value={result.p_value.toExponential(4)} />
              {result.effect_size !== null && result.effect_size !== undefined && (
                <StatCard title="Effect Size" value={result.effect_size.toFixed(4)} />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
