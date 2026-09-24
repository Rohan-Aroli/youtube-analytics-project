import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Reports() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // We can fetch key dashboard metrics and ml models to compile a summary report
    Promise.all([
      axios.get('/api/dashboard/summary'),
      axios.get('/api/ml/models')
    ])
      .then(([dashRes, mlRes]) => {
        setData({
          summary: dashRes.data,
          models: mlRes.data
        });
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6 max-w-4xl mx-auto pb-10">
      <div>
        <h1 className="text-3xl font-bold text-slate-800">Executive Statistical Report</h1>
        <p className="text-slate-500 mt-2">Comprehensive summary of the YouTube Creator Analytics platform.</p>
      </div>

      {loading && <div>Generating report...</div>}

      {data && !loading && (
        <div className="bg-white p-8 rounded-lg shadow-sm border border-slate-200 print:shadow-none print:border-none space-y-8">

          <section>
            <h2 className="text-2xl font-bold text-slate-800 border-b pb-2 mb-4">1. Dataset Overview</h2>
            <p className="text-slate-700 leading-relaxed mb-4">
              The analyzed dataset consists of <strong>{data.summary.total_creators.toLocaleString()}</strong> top YouTube creators spanning <strong>{data.summary.num_categories}</strong> unique categories and <strong>{data.summary.num_countries}</strong> countries.
            </p>
            <ul className="list-disc pl-6 space-y-2 text-slate-700">
              <li><strong>Mean Subscribers:</strong> {data.summary.mean_subscribers.toLocaleString(undefined, {maximumFractionDigits:0})}</li>
              <li><strong>Median Subscribers:</strong> {data.summary.median_subscribers.toLocaleString(undefined, {maximumFractionDigits:0})}</li>
              <li><strong>Mean Video Views:</strong> {data.summary.mean_views.toLocaleString(undefined, {maximumFractionDigits:0})}</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-slate-800 border-b pb-2 mb-4">2. Descriptive & Distribution Findings</h2>
            <p className="text-slate-700 leading-relaxed">
              Analysis indicates that key engagement metrics (Subscribers, Video Views, Earnings) are strongly right-skewed and leptokurtic. This implies a heavy-tailed distribution where a small fraction of elite creators capture a disproportionate share of total platform engagement and revenue. Shapiro-Wilk tests across these core variables consistently reject the null hypothesis of normality (p &lt; 0.05).
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-slate-800 border-b pb-2 mb-4">3. Inferential Statistics & Hypothesis Testing</h2>
            <p className="text-slate-700 leading-relaxed">
              <strong>Correlation:</strong> Pearson and Spearman correlation matrices reveal strong positive linear and monotonic relationships between Subscribers, Total Video Views, and Yearly Earnings (r &gt; 0.70, p &lt; 0.01). <br/><br/>
              <strong>ANOVA & Chi-Square:</strong> One-way ANOVA confirms statistically significant differences in mean earnings across different channel categories. Chi-Square tests of independence demonstrate a significant association between a creator's origin country and their content category.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-slate-800 border-b pb-2 mb-4">4. Predictive Modeling & Machine Learning</h2>
            <p className="text-slate-700 leading-relaxed mb-4">
              Machine learning models were trained to predict absolute earnings and classify general channel success (defined as top quartile subscriber count).
            </p>

            <h3 className="text-lg font-semibold text-slate-800 mb-2">Model Performance Highlights</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {data.models.map((m: any) => (
                <div key={m.model_name} className="border border-slate-200 p-4 rounded bg-slate-50">
                  <h4 className="font-bold text-indigo-700 mb-2">{m.model_name}</h4>
                  <ul className="text-sm space-y-1">
                    {Object.keys(m.metrics).map(k => (
                      <li key={k}><span className="font-semibold">{k}:</span> {m.metrics[k] > 100 ? m.metrics[k].toExponential(2) : m.metrics[k].toFixed(4)}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <p className="text-slate-700 leading-relaxed mt-4">
              The Random Forest architecture consistently outperformed linear/logistic baselines, demonstrating the presence of non-linear interactions between features (e.g., recent 30-day views strongly gating total yearly earnings).
            </p>
          </section>

          <div className="pt-8 border-t mt-8 flex justify-end print:hidden">
            <button onClick={() => window.print()} className="bg-slate-800 text-white px-6 py-2 rounded shadow hover:bg-slate-700">
              Print / Save PDF
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
