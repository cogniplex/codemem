import { useState, useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from 'recharts'
import { Calendar, Filter } from 'lucide-react'
import { useTimeline } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { SessionList } from './SessionList'

const TYPE_COLORS: Record<string, string> = {
  decision: '#8b5cf6',
  pattern: '#06b6d4',
  preference: '#f59e0b',
  style: '#ec4899',
  habit: '#10b981',
  insight: '#3b82f6',
  context: '#71717a',
}

export function TimelineView() {
  const namespace = useNamespaceStore((s) => s.active)
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')

  const params = useMemo(
    () => ({
      namespace: namespace ?? undefined,
      from: from || undefined,
      to: to || undefined,
    }),
    [namespace, from, to],
  )

  const { data: buckets, isLoading } = useTimeline(params)

  const memoryTypes = useMemo(() => {
    if (!buckets) return []
    const types = new Set<string>()
    for (const b of buckets) {
      for (const t of Object.keys(b.counts)) {
        types.add(t)
      }
    }
    return Array.from(types).sort()
  }, [buckets])

  const chartData = useMemo(() => {
    if (!buckets) return []
    return buckets.map((b) => ({
      date: b.date,
      ...b.counts,
      total: b.total,
    }))
  }, [buckets])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Calendar size={20} className="text-zinc-400" />
          <h2 className="text-lg font-semibold text-zinc-100">Timeline</h2>
        </div>
        <div className="flex items-center gap-3">
          <Filter size={14} className="text-zinc-500" />
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            From
            <input
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className="rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500 focus:outline-none"
            />
          </label>
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            To
            <input
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className="rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500 focus:outline-none"
            />
          </label>
          {(from || to) && (
            <button
              onClick={() => {
                setFrom('')
                setTo('')
              }}
              className="text-xs text-zinc-500 hover:text-zinc-300"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Stacked Area Chart */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-5">
        <h3 className="mb-4 text-sm font-medium text-zinc-300">
          Memories Over Time
        </h3>
        {isLoading ? (
          <div className="flex h-64 items-center justify-center">
            <span className="h-full w-full animate-pulse rounded bg-zinc-800" />
          </div>
        ) : !chartData.length ? (
          <div className="flex h-64 items-center justify-center text-sm text-zinc-500">
            No timeline data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart
              data={chartData}
              margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
            >
              <defs>
                {memoryTypes.map((type) => (
                  <linearGradient key={type} id={`gradient-${type}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={TYPE_COLORS[type] ?? '#52525b'} stopOpacity={0.4} />
                    <stop offset="100%" stopColor={TYPE_COLORS[type] ?? '#52525b'} stopOpacity={0.05} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#27272a"
                vertical={false}
              />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: '#71717a' }}
                tickLine={false}
                axisLine={{ stroke: '#3f3f46' }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#71717a' }}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid #3f3f46',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelStyle={{ color: '#d4d4d8' }}
                itemStyle={{ color: '#a1a1aa' }}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', color: '#a1a1aa' }}
              />
              {memoryTypes.map((type) => (
                <Area
                  key={type}
                  type="monotone"
                  dataKey={type}
                  stackId="memories"
                  stroke={TYPE_COLORS[type] ?? '#52525b'}
                  fill={`url(#gradient-${type})`}
                  strokeWidth={1.5}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Session List */}
      <SessionList />
    </div>
  )
}
