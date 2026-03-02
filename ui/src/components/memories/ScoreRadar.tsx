import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts'
import type { ScoreBreakdown } from '../../api/types'

const labels: { key: keyof ScoreBreakdown; label: string }[] = [
  { key: 'vector_similarity', label: 'Vector' },
  { key: 'graph_strength', label: 'Graph' },
  { key: 'token_overlap', label: 'Token' },
  { key: 'temporal', label: 'Temporal' },
  { key: 'tag_matching', label: 'Tags' },
  { key: 'importance', label: 'Importance' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'recency', label: 'Recency' },
]

export function ScoreRadar({ breakdown }: { breakdown: ScoreBreakdown }) {
  const data = labels.map(({ key, label }) => ({
    axis: label,
    value: breakdown[key] ?? 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#3f3f46" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        <Radar
          dataKey="value"
          stroke="#8b5cf6"
          fill="#8b5cf6"
          fillOpacity={0.25}
          strokeWidth={1.5}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
