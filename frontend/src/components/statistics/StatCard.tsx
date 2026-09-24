import React from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
}

export default function StatCard({ title, value, subtitle, icon }: StatCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5 flex flex-col">
      <div className="flex justify-between items-start">
        <h3 className="text-slate-500 text-sm font-medium">{title}</h3>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
      <div className="mt-2 flex-1">
        <p className="text-2xl font-semibold text-slate-800">{value}</p>
        {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
      </div>
    </div>
  );
}
