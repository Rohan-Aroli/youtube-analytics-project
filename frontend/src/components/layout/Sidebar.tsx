import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  TableProperties,
  BarChart2,
  Activity,
  GitMerge,
  TestTube,
  TrendingUp,
  Brain,
  Users,
  FileText
} from 'lucide-react';

const navItems = [
  { name: 'Dashboard', path: '/', icon: <LayoutDashboard size={20} /> },
  { name: 'Dataset Explorer', path: '/dataset', icon: <TableProperties size={20} /> },
  { name: 'Creator Comparison', path: '/compare', icon: <Users size={20} /> },
  { name: 'Descriptive Stats', path: '/descriptive', icon: <BarChart2 size={20} /> },
  { name: 'Distributions', path: '/distributions', icon: <Activity size={20} /> },
  { name: 'Correlation', path: '/correlation', icon: <GitMerge size={20} /> },
  { name: 'Hypothesis Testing', path: '/hypothesis', icon: <TestTube size={20} /> },
  { name: 'Regression Analysis', path: '/regression', icon: <TrendingUp size={20} /> },
  { name: 'Machine Learning', path: '/ml', icon: <Brain size={20} /> },
  { name: 'Statistical Report', path: '/report', icon: <FileText size={20} /> },
];

export default function Sidebar() {
  return (
    <div className="flex flex-col w-64 bg-slate-900 text-white min-h-screen border-r border-slate-700">
      <div className="p-5 font-bold text-xl border-b border-slate-700">
        YT Analytics Pro
      </div>
      <nav className="flex-1 overflow-y-auto py-4">
        <ul className="space-y-1 px-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors ${
                    isActive ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`
                }
              >
                {item.icon}
                <span className="text-sm font-medium">{item.name}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
