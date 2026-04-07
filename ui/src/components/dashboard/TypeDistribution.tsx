import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { useDistribution } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'

const TYPE_COLORS: Record<string, string> = {
  decision: '#8b5cf6',
  pattern: '#22d3ee',
  preference: '#f59e0b',
  style: '#ec4899',
  habit: '#10b981',
  insight: '#3b82f6',
  context: '#71717a',
}

export function TypeDistribution() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useDistribution(namespace ?? undefined)

  const chartData = Object.entries(data?.type_counts ?? {}).map(([name, value]) => ({
    name,
    value,
  }))

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-900">
      <div className="flex items-center gap-2 border-b border-zinc-800/60 px-5 py-3">
        <h3 className="text-[13px] font-medium text-zinc-100">Type Distribution</h3>
      </div>
      <div className="flex items-center justify-center px-5 py-4">
        {isLoading ? (
          <div className="flex h-[200px] w-full items-center justify-center">
            <span className="h-32 w-32 animate-pulse rounded-full bg-zinc-800" />
          </div>
        ) : chartData.length === 0 ? (
          <p className="py-12 text-sm text-zinc-500">No data yet</p>
        ) : (
          <div className="flex w-full items-center gap-6">
            <div className="h-[200px] w-[200px] shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={85}
                    paddingAngle={2}
                    strokeWidth={0}
                  >
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={TYPE_COLORS[entry.name] ?? TYPE_COLORS.context}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#27272a',
                      border: '1px solid #3f3f46',
                      borderRadius: '0.5rem',
                      fontSize: '0.75rem',
                    }}
                    itemStyle={{ color: '#d4d4d8' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col gap-2">
              {chartData.map((entry) => (
                <div key={entry.name} className="flex items-center gap-2 text-xs">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{
                      backgroundColor: TYPE_COLORS[entry.name] ?? TYPE_COLORS.context,
                    }}
                  />
                  <span className="text-zinc-400">{entry.name}</span>
                  <span className="ml-auto tabular-nums text-zinc-500">{entry.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
