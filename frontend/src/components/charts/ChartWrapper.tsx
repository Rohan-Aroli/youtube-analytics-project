import React from 'react';

interface ChartWrapperProps {
  title: string;
  description?: string;
  children: React.ReactNode;
}

export default function ChartWrapper({ title, description, children }: ChartWrapperProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5 w-full">
      <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
      {description && <p className="text-sm text-slate-500 mb-4">{description}</p>}
      <div className="w-full h-80 mt-4">
        {children}
      </div>
    </div>
  );
}
