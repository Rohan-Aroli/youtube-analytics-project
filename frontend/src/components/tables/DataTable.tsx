

interface Column {
  header: string;
  accessor: string;
}

interface DataTableProps {
  columns: Column[];
  data: any[];
}

export default function DataTable({ columns, data }: DataTableProps) {
  return (
    <div className="overflow-x-auto w-full bg-white rounded-lg shadow-sm border border-slate-200">
      <table className="w-full text-sm text-left text-slate-600">
        <thead className="text-xs text-slate-700 uppercase bg-slate-50 border-b border-slate-200">
          <tr>
            {columns.map((col, i) => (
              <th key={i} className="px-6 py-3 whitespace-nowrap">{col.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="bg-white border-b hover:bg-slate-50">
              {columns.map((col, j) => (
                <td key={j} className="px-6 py-4">
                  {row[col.accessor] !== null && row[col.accessor] !== undefined
                    ? String(row[col.accessor])
                    : '-'}
                </td>
              ))}
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-6 py-4 text-center text-slate-500">
                No data available
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
