import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';

// Pages
import Dashboard from './pages/Dashboard';
import DatasetExplorer from './pages/DatasetExplorer';
import CreatorComparison from './pages/CreatorComparison';
import DescriptiveStatistics from './pages/DescriptiveStatistics';
import Distributions from './pages/Distributions';
import Correlation from './pages/Correlation';
import HypothesisTesting from './pages/HypothesisTesting';
import Regression from './pages/Regression';
import MachineLearning from './pages/MachineLearning';
import Reports from './pages/Reports';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="dataset" element={<DatasetExplorer />} />
          <Route path="compare" element={<CreatorComparison />} />
          <Route path="descriptive" element={<DescriptiveStatistics />} />
          <Route path="distributions" element={<Distributions />} />
          <Route path="correlation" element={<Correlation />} />
          <Route path="hypothesis" element={<HypothesisTesting />} />
          <Route path="regression" element={<Regression />} />
          <Route path="ml" element={<MachineLearning />} />
          <Route path="report" element={<Reports />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
