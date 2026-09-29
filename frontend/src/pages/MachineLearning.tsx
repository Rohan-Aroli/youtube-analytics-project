import { useState, useEffect } from 'react';
import axios from 'axios';
import StatCard from '../components/statistics/StatCard';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import ChartWrapper from '../components/charts/ChartWrapper';

export default function MachineLearning() {
  const [models, setModels] = useState<any[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  // Prediction State
  const [selectedModel, setSelectedModel] = useState("Random Forest Regressor");
  const [features, setFeatures] = useState({
    subscribers: 10000000,
    video_views: 5000000000,
    uploads: 500,
    category: "Entertainment",
    country: "United States",
    subscribers_for_last_30_days: 50000,
    video_views_for_the_last_30_days: 10000000
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoadingModels(true);
    axios.get('/api/ml/models')
      .then(res => {
        setModels(res.data);
      })
      .catch(err => console.error(err))
      .finally(() => setLoadingModels(false));
  }, []);

  const handlePredict = () => {
    setPredicting(true);
    setError("");
    setPrediction(null);

    const taskType = models.find(m => m.model_name === selectedModel)?.task_type;
    const endpoint = taskType === 'classification' ? 'classify' : 'predict';

    axios.post(`/api/ml/${endpoint}`, {
      model_name: selectedModel,
      features: features
    })
    .then(res => setPrediction(res.data))
    .catch(err => setError(err.response?.data?.detail || "Prediction failed"))
    .finally(() => setPredicting(false));
  };

  const currentModelData = models.find(m => m.model_name === selectedModel);

  const formatFeatureImportance = (importances: any) => {
    if (!importances) return [];
    return Object.keys(importances).map(k => ({
      name: k.replace('country_', '').replace('category_', ''),
      value: importances[k]
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Machine Learning Intelligence</h1>
        <p className="text-slate-500">Predict earnings and classify creator success using trained models.</p>
      </div>

      {loadingModels ? (
        <div>Loading models...</div>
      ) : (
        <>
          <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
            <div className="flex gap-4 mb-6">
              {models.map(m => (
                <button
                  key={m.model_name}
                  onClick={() => setSelectedModel(m.model_name)}
                  className={`px-4 py-2 rounded-md ${selectedModel === m.model_name ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}
                >
                  {m.model_name}
                </button>
              ))}
            </div>

            {currentModelData && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.keys(currentModelData.metrics).map(metric => (
                  <StatCard
                    key={metric}
                    title={metric}
                    value={currentModelData.metrics[metric] > 100 ? currentModelData.metrics[metric].toExponential(4) : currentModelData.metrics[metric].toFixed(4)}
                  />
                ))}
              </div>
            )}

            {currentModelData?.feature_importance && (
              <div className="mt-8">
                <ChartWrapper title="Feature Importance" description={`Top features driving the ${selectedModel} model.`}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={formatFeatureImportance(currentModelData.feature_importance)} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={150} tick={{fontSize: 12}} />
                      <RechartsTooltip />
                      <Bar dataKey="value" fill="#6366f1" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartWrapper>
              </div>
            )}
          </div>

          <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4">Make a Prediction</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {Object.keys(features).map(key => (
                <div key={key}>
                  <label className="block text-sm font-medium text-slate-700 mb-1 capitalize">{key.replace(/_/g, ' ')}</label>
                  <input
                    type={typeof features[key as keyof typeof features] === 'number' ? 'number' : 'text'}
                    value={features[key as keyof typeof features]}
                    onChange={(e) => setFeatures({...features, [key]: typeof features[key as keyof typeof features] === 'number' ? Number(e.target.value) : e.target.value})}
                    className="w-full border rounded p-2 focus:border-indigo-500 focus:ring-indigo-500"
                  />
                </div>
              ))}
            </div>

            <button
              onClick={handlePredict}
              disabled={predicting}
              className="bg-emerald-600 text-white px-6 py-2 rounded-md hover:bg-emerald-700 disabled:opacity-50"
            >
              {predicting ? 'Predicting...' : 'Predict'}
            </button>

            {error && <div className="text-red-500 mt-4">{error}</div>}

            {prediction && (
              <div className="mt-6 p-5 bg-slate-50 border border-slate-200 rounded-lg text-center">
                <h3 className="text-lg font-semibold text-slate-700 mb-2">Prediction Result</h3>
                {prediction.predicted_class ? (
                  <>
                    <p className="text-3xl font-bold text-indigo-600 mb-1">{prediction.predicted_class}</p>
                    <p className="text-sm text-slate-500">Probability: {(prediction.probability * 100).toFixed(2)}%</p>
                  </>
                ) : (
                  <p className="text-3xl font-bold text-emerald-600">${prediction.prediction.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
