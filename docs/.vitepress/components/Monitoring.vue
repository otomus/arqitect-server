<template>
  <div class="mon">
    <!-- Connection bar -->
    <div class="mon-connect">
      <div class="mon-connect-left">
        <span class="mon-dot" :class="connected ? 'on' : 'off'"></span>
        <span class="mon-label">{{ connected ? 'CONNECTED' : 'DISCONNECTED' }}</span>
      </div>
      <div class="mon-connect-mid">
        <input
          v-model="endpoint"
          class="mon-input"
          placeholder="http://localhost:7681"
          @keyup.enter="connect"
        />
        <button class="mon-btn" @click="connect">Connect</button>
      </div>
      <div class="mon-connect-right">
        <button class="mon-btn mon-btn-ghost" @click="refresh" :disabled="!connected">
          Refresh
        </button>
      </div>
    </div>

    <!-- Empty state -->
    <div v-if="!connected" class="mon-empty">
      <div class="mon-empty-icon">&#9678;</div>
      <h2>Arqitect Monitoring</h2>
      <p>Connect to your local arqitect server to view traces.</p>
      <div class="mon-instructions">
        <code>make start</code> launches the trace server on <code>localhost:7681</code>
      </div>
    </div>

    <!-- Dashboard -->
    <div v-else>
      <!-- KPI cards -->
      <div class="mon-kpis">
        <div class="mon-kpi">
          <div class="mon-kpi-value">{{ stats.total_spans?.toLocaleString() || 0 }}</div>
          <div class="mon-kpi-label">Total Spans</div>
        </div>
        <div class="mon-kpi">
          <div class="mon-kpi-value mon-kpi-error">{{ stats.total_errors || 0 }}</div>
          <div class="mon-kpi-label">Errors</div>
        </div>
        <div class="mon-kpi">
          <div class="mon-kpi-value">{{ stats.error_rate || 0 }}%</div>
          <div class="mon-kpi-label">Error Rate</div>
        </div>
        <div class="mon-kpi">
          <div class="mon-kpi-value">{{ stats.total_files || 0 }}</div>
          <div class="mon-kpi-label">Trace Sessions</div>
        </div>
      </div>

      <!-- Latency table by type -->
      <div class="mon-section" v-if="stats.by_type">
        <h3 class="mon-section-title">Latency by Operation</h3>
        <div class="mon-table-wrap">
          <table class="mon-table">
            <thead>
              <tr>
                <th>Operation</th>
                <th>Count</th>
                <th>Avg</th>
                <th>p50</th>
                <th>p95</th>
                <th>p99</th>
                <th>Max</th>
                <th class="mon-bar-col">Distribution</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(data, type) in stats.by_type" :key="type">
                <td><span class="mon-tag" :class="'mon-tag-' + type">{{ type }}</span></td>
                <td class="mon-mono">{{ data.count.toLocaleString() }}</td>
                <td class="mon-mono">{{ data.avg_ms }}ms</td>
                <td class="mon-mono">{{ data.p50_ms }}ms</td>
                <td class="mon-mono">{{ data.p95_ms }}ms</td>
                <td class="mon-mono">{{ data.p99_ms }}ms</td>
                <td class="mon-mono">{{ data.max_ms }}ms</td>
                <td class="mon-bar-col">
                  <div class="mon-latency-bar">
                    <div
                      class="mon-latency-fill"
                      :class="'mon-fill-' + type"
                      :style="{ width: barWidth(data.avg_ms) }"
                    ></div>
                    <div
                      class="mon-latency-p95"
                      :style="{ left: barWidth(data.p95_ms) }"
                    ></div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Trace file picker + waterfall -->
      <div class="mon-section">
        <div class="mon-section-header">
          <h3 class="mon-section-title">Trace Explorer</h3>
          <select v-model="selectedFile" class="mon-select" @change="loadTrace">
            <option value="">Select trace...</option>
            <option v-for="f in files" :key="f.name" :value="f.name">
              {{ f.name }} ({{ f.size }})
            </option>
          </select>
        </div>

        <!-- Filters -->
        <div v-if="spans.length" class="mon-filters">
          <div class="mon-filter-row">
            <div class="mon-filter-group">
              <label class="mon-filter-label">Search</label>
              <input
                v-model="filterText"
                class="mon-input mon-input-sm"
                placeholder="span name or attribute..."
              />
            </div>
            <div class="mon-filter-group">
              <label class="mon-filter-label">Type</label>
              <div class="mon-filter-pills">
                <button
                  v-for="t in allTypes"
                  :key="t"
                  class="mon-pill"
                  :class="{ active: filterTypes.has(t), ['mon-pill-' + t]: true }"
                  @click="toggleType(t)"
                >{{ t }}</button>
              </div>
            </div>
            <div class="mon-filter-group">
              <label class="mon-filter-label">Status</label>
              <div class="mon-filter-pills">
                <button
                  class="mon-pill"
                  :class="{ active: filterStatus === 'all' }"
                  @click="filterStatus = 'all'"
                >All</button>
                <button
                  class="mon-pill mon-pill-err"
                  :class="{ active: filterStatus === 'error' }"
                  @click="filterStatus = 'error'"
                >Errors</button>
                <button
                  class="mon-pill mon-pill-ok"
                  :class="{ active: filterStatus === 'ok' }"
                  @click="filterStatus = 'ok'"
                >OK</button>
              </div>
            </div>
            <div class="mon-filter-group">
              <label class="mon-filter-label">Duration (ms)</label>
              <div class="mon-filter-range">
                <input
                  v-model.number="filterMinMs"
                  type="number"
                  class="mon-input mon-input-xs"
                  placeholder="min"
                  min="0"
                />
                <span class="mon-filter-dash">&ndash;</span>
                <input
                  v-model.number="filterMaxMs"
                  type="number"
                  class="mon-input mon-input-xs"
                  placeholder="max"
                  min="0"
                />
              </div>
            </div>
            <div class="mon-filter-group mon-filter-actions">
              <button class="mon-btn mon-btn-ghost mon-btn-xs" @click="clearFilters">Clear</button>
              <span class="mon-filter-count">{{ filteredFlatTree.length }} / {{ flatTree.length }} spans</span>
            </div>
          </div>
        </div>

        <!-- Waterfall -->
        <div v-if="spans.length" class="mon-waterfall">
          <div class="mon-waterfall-header">
            <span class="mon-wf-col-name">Operation</span>
            <span class="mon-wf-col-svc">Type</span>
            <span class="mon-wf-col-dur mon-sortable" @click="setSort('duration')">
              Duration {{ sortIndicator('duration') }}
            </span>
            <span class="mon-wf-col-time mon-sortable" @click="setSort('start')">
              Start {{ sortIndicator('start') }}
            </span>
            <span class="mon-wf-col-time mon-sortable" @click="setSort('end')">
              End {{ sortIndicator('end') }}
            </span>
          </div>
          <div
            v-for="(node, idx) in filteredFlatTree"
            :key="node.span_id"
            class="mon-wf-row"
            :class="{ 'mon-wf-error': node.status === 'ERROR', 'mon-wf-selected': selectedSpan?.span_id === node.span_id }"
            @click="selectSpan(node)"
          >
            <span class="mon-wf-col-name">
              <span :style="{ paddingLeft: node._depth * 16 + 'px' }">
                <span v-if="node._hasChildren" class="mon-wf-toggle" @click.stop="toggleNode(node)">
                  {{ node._collapsed ? '&#9656;' : '&#9662;' }}
                </span>
                <span v-else class="mon-wf-dot">&#8226;</span>
                {{ node.name }}
              </span>
            </span>
            <span class="mon-wf-col-svc">
              <span class="mon-tag mon-tag-sm" :class="'mon-tag-' + categoryOf(node.name)">
                {{ categoryOf(node.name) }}
              </span>
            </span>
            <span class="mon-wf-col-dur mon-mono">{{ node.duration_ms }}ms</span>
            <span class="mon-wf-col-time mon-mono">{{ formatNs(node.start_time_ns) }}</span>
            <span class="mon-wf-col-time mon-mono">{{ formatNs(node.end_time_ns) }}</span>
          </div>
        </div>
      </div>

      <!-- Detail panel (slide-out) -->
      <Teleport to="body">
        <div class="mon-detail-overlay" v-if="selectedSpan" @click.self="selectedSpan = null">
          <div class="mon-detail">
            <div class="mon-detail-header">
              <h3>{{ selectedSpan.name }}</h3>
              <button class="mon-btn mon-btn-ghost" @click="selectedSpan = null">&#10005;</button>
            </div>
            <div class="mon-detail-body">
              <div class="mon-detail-section">
                <h4>Timing</h4>
                <dl class="mon-dl">
                  <dt>Duration</dt><dd>{{ selectedSpan.duration_ms }} ms</dd>
                  <dt>Status</dt>
                  <dd>
                    <span class="mon-tag mon-tag-sm" :class="selectedSpan.status === 'ERROR' ? 'mon-tag-error' : 'mon-tag-ok'">
                      {{ selectedSpan.status }}
                    </span>
                  </dd>
                  <dt>Trace ID</dt><dd class="mon-mono">{{ selectedSpan.trace_id }}</dd>
                  <dt>Span ID</dt><dd class="mon-mono">{{ selectedSpan.span_id }}</dd>
                  <dt>Parent</dt><dd class="mon-mono">{{ selectedSpan.parent_span_id || '(root)' }}</dd>
                </dl>
              </div>
              <div class="mon-detail-section" v-if="attrKeys(selectedSpan).length">
                <h4>Attributes</h4>
                <dl class="mon-dl">
                  <template v-for="k in attrKeys(selectedSpan)" :key="k">
                    <dt>{{ k }}</dt>
                    <dd class="mon-mono mon-dd-val">{{ selectedSpan.attributes[k] }}</dd>
                  </template>
                </dl>
              </div>
              <div class="mon-detail-section" v-if="selectedSpan.events?.length">
                <h4>Events ({{ selectedSpan.events.length }})</h4>
                <div v-for="(ev, i) in selectedSpan.events" :key="i" class="mon-event">
                  <strong>{{ ev.name }}</strong>
                  <dl class="mon-dl" v-if="Object.keys(ev.attributes || {}).length">
                    <template v-for="(v, k) in ev.attributes" :key="k">
                      <dt>{{ k }}</dt><dd class="mon-mono mon-dd-val">{{ v }}</dd>
                    </template>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Teleport>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const endpoint = ref('http://localhost:7681')
const connected = ref(false)
const stats = ref({})
const files = ref([])
const spans = ref([])
const selectedFile = ref('')
const selectedSpan = ref(null)
const collapsedIds = ref(new Set())
const filterText = ref('')
const filterTypes = ref(new Set())
const filterStatus = ref('all')
const filterMinMs = ref(null)
const filterMaxMs = ref(null)
const allTypes = ['llm', 'nerve', 'brain', 'dreamstate', 'synthesis', 'other']
const sortField = ref(null)
const sortAsc = ref(true)

async function connect() {
  try {
    const resp = await fetch(endpoint.value + '/api/health')
    if (resp.ok) {
      connected.value = true
      await refresh()
    }
  } catch {
    connected.value = false
    alert('Cannot connect to ' + endpoint.value + '. Is the server running? (make start)')
  }
}

async function refresh() {
  const [statsResp, filesResp] = await Promise.all([
    fetch(endpoint.value + '/api/stats'),
    fetch(endpoint.value + '/api/files'),
  ])
  stats.value = await statsResp.json()
  files.value = await filesResp.json()
  if (files.value.length && !selectedFile.value) {
    selectedFile.value = files.value[0].name
    await loadTrace()
  }
}

async function loadTrace() {
  if (!selectedFile.value) return
  const resp = await fetch(endpoint.value + '/api/trace?file=' + encodeURIComponent(selectedFile.value))
  spans.value = await resp.json()
  collapsedIds.value = new Set()
  selectedSpan.value = null
}

function categoryOf(name) {
  if (name.startsWith('llm.')) return 'llm'
  if (name.startsWith('nerve.synth') || name.startsWith('synth.')) return 'synthesis'
  if (name.startsWith('nerve.')) return 'nerve'
  if (name.startsWith('dreamstate.')) return 'dreamstate'
  if (name.startsWith('brain.')) return 'brain'
  return 'other'
}

function buildTree(spans) {
  const byId = {}
  const roots = []
  spans.forEach(s => { byId[s.span_id] = { ...s, children: [] } })
  spans.forEach(s => {
    const node = byId[s.span_id]
    if (s.parent_span_id && byId[s.parent_span_id]) {
      byId[s.parent_span_id].children.push(node)
    } else {
      roots.push(node)
    }
  })
  return roots
}

function flattenTree(nodes, depth = 0) {
  const result = []
  for (const n of nodes.sort((a, b) => a.start_time_ns - b.start_time_ns)) {
    result.push({ ...n, _depth: depth, _hasChildren: n.children.length > 0, _collapsed: collapsedIds.value.has(n.span_id) })
    if (n.children.length && !collapsedIds.value.has(n.span_id)) {
      result.push(...flattenTree(n.children, depth + 1))
    }
  }
  return result
}

const flatTree = computed(() => {
  if (!spans.value.length) return []
  return flattenTree(buildTree(spans.value))
})

const timeRange = computed(() => {
  if (!spans.value.length) return { min: 0, max: 1 }
  const min = Math.min(...spans.value.map(s => s.start_time_ns))
  const max = Math.max(...spans.value.map(s => s.end_time_ns))
  return { min, max: max === min ? min + 1 : max }
})

function barStyle(node) {
  const { min, max } = timeRange.value
  const range = max - min
  const left = ((node.start_time_ns - min) / range) * 100
  const width = Math.max(0.3, ((node.end_time_ns - node.start_time_ns) / range) * 100)
  return { left: left + '%', width: width + '%' }
}

function barWidth(ms) {
  const maxMs = Math.max(...Object.values(stats.value.by_type || {}).map(d => d.max_ms || 0), 1)
  return Math.min(100, (ms / maxMs) * 100) + '%'
}

function toggleNode(node) {
  if (collapsedIds.value.has(node.span_id)) {
    collapsedIds.value.delete(node.span_id)
  } else {
    collapsedIds.value.add(node.span_id)
  }
  collapsedIds.value = new Set(collapsedIds.value)
}

function toggleType(t) {
  if (filterTypes.value.has(t)) {
    filterTypes.value.delete(t)
  } else {
    filterTypes.value.add(t)
  }
  filterTypes.value = new Set(filterTypes.value)
}

function clearFilters() {
  filterText.value = ''
  filterTypes.value = new Set()
  filterStatus.value = 'all'
  filterMinMs.value = null
  filterMaxMs.value = null
}

function matchesFilter(node) {
  if (filterTypes.value.size && !filterTypes.value.has(categoryOf(node.name))) return false
  if (filterStatus.value === 'error' && node.status !== 'ERROR') return false
  if (filterStatus.value === 'ok' && node.status === 'ERROR') return false
  if (filterMinMs.value != null && node.duration_ms < filterMinMs.value) return false
  if (filterMaxMs.value != null && node.duration_ms > filterMaxMs.value) return false
  if (filterText.value) {
    const q = filterText.value.toLowerCase()
    const inName = node.name.toLowerCase().includes(q)
    const inAttrs = Object.entries(node.attributes || {}).some(
      ([k, v]) => k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q)
    )
    if (!inName && !inAttrs) return false
  }
  return true
}

function setSort(field) {
  if (sortField.value === field) {
    if (!sortAsc.value) {
      sortField.value = null
      sortAsc.value = true
    } else {
      sortAsc.value = false
    }
  } else {
    sortField.value = field
    sortAsc.value = true
  }
}

function sortIndicator(field) {
  if (sortField.value !== field) return '\u2195'
  return sortAsc.value ? '\u25B2' : '\u25BC'
}

function sortKey(node) {
  if (sortField.value === 'duration') return node.duration_ms
  if (sortField.value === 'end') return node.end_time_ns
  return node.start_time_ns
}

const filteredFlatTree = computed(() => {
  const filtered = flatTree.value.filter(matchesFilter)
  if (!sortField.value) return filtered
  const dir = sortAsc.value ? 1 : -1
  return [...filtered].sort((a, b) => (sortKey(a) - sortKey(b)) * dir)
})

function formatNs(ns) {
  const d = new Date(ns / 1e6)
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  const s = String(d.getSeconds()).padStart(2, '0')
  const ms = String(d.getMilliseconds()).padStart(3, '0')
  return `${h}:${m}:${s}.${ms}`
}

function selectSpan(node) { selectedSpan.value = node }
function attrKeys(span) { return Object.keys(span.attributes || {}) }

onMounted(() => { connect() })
</script>

<style scoped>
/* ---- Layout ---- */
.mon { max-width: 1400px; margin: 0 auto; padding: 120px 1rem 4rem; }

/* ---- Connection bar ---- */
.mon-connect {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; margin-bottom: 24px;
  background: var(--arq-bg-card); border: 1px solid var(--arq-border);
  border-radius: 6px;
}
.mon-connect-left { display: flex; align-items: center; gap: 8px; min-width: 140px; }
.mon-connect-mid { display: flex; gap: 8px; flex: 1; }
.mon-connect-right { margin-left: auto; }
.mon-dot {
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}
.mon-dot.on { background: var(--arq-teal); box-shadow: 0 0 8px var(--arq-teal); }
.mon-dot.off { background: #f85149; box-shadow: 0 0 8px rgba(248,81,73,0.5); }
.mon-label {
  font-family: var(--arq-font-display); font-size: 0.65rem;
  letter-spacing: 0.15em; color: var(--arq-text-secondary);
}
.mon-input {
  background: var(--arq-bg-primary); color: var(--arq-text-primary);
  border: 1px solid var(--arq-border); border-radius: 4px;
  padding: 6px 12px; font-family: var(--arq-font-mono); font-size: 0.8rem;
  width: 280px; outline: none; transition: border-color 0.2s;
}
.mon-input:focus { border-color: var(--arq-cyan); }
.mon-btn {
  background: var(--arq-cyan); color: #050510; border: none;
  border-radius: 4px; padding: 6px 16px; cursor: pointer;
  font-family: var(--arq-font-display); font-size: 0.65rem;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  transition: all 0.2s;
}
.mon-btn:hover { box-shadow: 0 0 12px var(--arq-cyan-glow); }
.mon-btn-ghost {
  background: transparent; color: var(--arq-text-secondary);
  border: 1px solid var(--arq-border);
}
.mon-btn-ghost:hover { border-color: var(--arq-cyan); color: var(--arq-cyan); }

/* ---- Empty state ---- */
.mon-empty { text-align: center; padding: 120px 20px; }
.mon-empty-icon {
  font-size: 48px; color: var(--arq-cyan); opacity: 0.3;
  margin-bottom: 16px;
}
.mon-empty h2 {
  font-family: var(--arq-font-display); color: var(--arq-cyan);
  font-size: 1.2rem; letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 8px;
}
.mon-empty p { color: var(--arq-text-secondary); font-size: 0.85rem; }
.mon-instructions {
  margin-top: 24px; color: var(--arq-text-dim); font-size: 0.8rem;
}
.mon-instructions code {
  background: var(--arq-bg-card); border: 1px solid var(--arq-border);
  padding: 2px 8px; border-radius: 3px; color: var(--arq-teal);
}

/* ---- KPI cards ---- */
.mon-kpis {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  margin-bottom: 24px;
}
.mon-kpi {
  background: var(--arq-bg-card); border: 1px solid var(--arq-border);
  border-radius: 6px; padding: 20px; text-align: center;
  transition: border-color 0.2s;
}
.mon-kpi:hover { border-color: var(--arq-border-bright); }
.mon-kpi-value {
  font-family: var(--arq-font-display); font-size: 1.8rem;
  font-weight: 700; color: var(--arq-cyan);
  text-shadow: 0 0 10px var(--arq-cyan-dim);
}
.mon-kpi-error { color: #f85149 !important; text-shadow: 0 0 10px rgba(248,81,73,0.3) !important; }
.mon-kpi-label {
  font-family: var(--arq-font-display); font-size: 0.6rem;
  text-transform: uppercase; letter-spacing: 0.15em;
  color: var(--arq-text-dim); margin-top: 6px;
}

/* ---- Sections ---- */
.mon-section {
  background: var(--arq-bg-card); border: 1px solid var(--arq-border);
  border-radius: 6px; padding: 20px; margin-bottom: 24px;
}
.mon-section-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.mon-section-title {
  font-family: var(--arq-font-display); font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.15em;
  color: var(--arq-cyan); margin-bottom: 16px;
}
.mon-section-header .mon-section-title { margin-bottom: 0; }
.mon-select {
  background: var(--arq-bg-primary); color: var(--arq-text-primary);
  border: 1px solid var(--arq-border); border-radius: 4px;
  padding: 6px 12px; font-family: var(--arq-font-mono); font-size: 0.75rem;
  min-width: 300px;
}

/* ---- Tags ---- */
.mon-tag {
  display: inline-block; padding: 3px 10px; border-radius: 3px;
  font-family: var(--arq-font-display); font-size: 0.7rem;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
}
.mon-tag-sm { padding: 2px 8px; font-size: 0.65rem; }
.mon-tag-llm { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
.mon-tag-nerve { background: var(--arq-teal-dim); color: var(--arq-teal); border: 1px solid rgba(0,255,136,0.3); }
.mon-tag-brain { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.mon-tag-synthesis { background: rgba(219,109,40,0.15); color: #db6d28; border: 1px solid rgba(219,109,40,0.3); }
.mon-tag-dreamstate { background: rgba(188,143,255,0.15); color: #bc8fff; border: 1px solid rgba(188,143,255,0.3); }
.mon-tag-other { background: var(--arq-bg-secondary); color: var(--arq-text-dim); border: 1px solid var(--arq-border); }
.mon-tag-error { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.mon-tag-ok { background: var(--arq-teal-dim); color: var(--arq-teal); border: 1px solid rgba(0,255,136,0.3); }

/* ---- Latency table ---- */
.mon-table-wrap { overflow-x: auto; }
.mon-table {
  width: 100%; border-collapse: collapse; font-size: 0.9rem;
}
.mon-table th {
  font-family: var(--arq-font-display); font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--arq-text-secondary); text-align: left;
  padding: 10px 12px; border-bottom: 1px solid var(--arq-border);
  font-weight: 700;
}
.mon-table td {
  padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.03);
  color: var(--arq-text-primary); font-weight: 500;
}
.mon-table tr:hover td { background: rgba(0,212,255,0.03); }
.mon-mono { font-family: var(--arq-font-mono); font-size: 0.85rem; font-weight: 500; }
.mon-bar-col { min-width: 200px; }
.mon-latency-bar {
  height: 6px; background: var(--arq-bg-primary); border-radius: 3px;
  position: relative; overflow: visible;
}
.mon-latency-fill {
  height: 100%; border-radius: 3px; transition: width 0.5s ease;
}
.mon-fill-llm { background: linear-gradient(90deg, rgba(88,166,255,0.6), #58a6ff); }
.mon-fill-nerve { background: linear-gradient(90deg, rgba(0,255,136,0.6), var(--arq-teal)); }
.mon-fill-brain { background: linear-gradient(90deg, rgba(210,153,34,0.6), #d29922); }
.mon-fill-synthesis { background: linear-gradient(90deg, rgba(219,109,40,0.6), #db6d28); }
.mon-fill-dreamstate { background: linear-gradient(90deg, rgba(188,143,255,0.6), #bc8fff); }
.mon-fill-other { background: var(--arq-border-bright); }
.mon-latency-p95 {
  position: absolute; top: -3px; width: 2px; height: 12px;
  background: var(--arq-text-dim); border-radius: 1px;
}

/* ---- Waterfall ---- */
.mon-waterfall { font-size: 0.9rem; }
.mon-waterfall-header {
  display: flex; padding: 8px 12px; gap: 12px;
  font-family: var(--arq-font-display); font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--arq-text-secondary); border-bottom: 1px solid var(--arq-border);
  font-weight: 700;
}
.mon-wf-row {
  display: flex; align-items: center; padding: 4px 12px; gap: 12px;
  cursor: pointer; transition: background 0.1s;
  border-left: 2px solid transparent;
}
.mon-wf-row:hover { background: rgba(0,212,255,0.03); }
.mon-wf-row.mon-wf-error { border-left-color: #f85149; }
.mon-wf-row.mon-wf-selected { background: rgba(0,212,255,0.08); border-left-color: var(--arq-cyan); }
.mon-wf-col-name { width: 300px; min-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
.mon-wf-col-svc { width: 110px; min-width: 110px; }
.mon-wf-col-dur { width: 100px; min-width: 100px; text-align: right; color: var(--arq-text-primary); font-weight: 600; }
.mon-sortable {
  cursor: pointer; user-select: none; transition: color 0.15s;
}
.mon-sortable:hover { color: var(--arq-cyan); }
.mon-wf-col-time {
  width: 110px; min-width: 110px; text-align: right;
  color: var(--arq-text-primary); font-size: 0.8rem;
  font-weight: 600;
}
.mon-wf-toggle {
  display: inline-block; width: 16px; text-align: center; cursor: pointer;
  color: var(--arq-text-dim); font-size: 10px;
}
.mon-wf-dot {
  display: inline-block; width: 16px; text-align: center;
  color: var(--arq-border-bright); font-size: 8px;
}

/* ---- Detail panel ---- */
.mon-detail-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  z-index: 10000; display: flex; justify-content: flex-end;
}
.mon-detail {
  width: 560px; max-width: 90vw; height: 100vh; background: var(--arq-bg-primary);
  border-left: 1px solid var(--arq-border); overflow-y: auto;
  animation: mon-slide-in 0.2s ease;
}
@keyframes mon-slide-in { from { transform: translateX(100%); } to { transform: translateX(0); } }
.mon-detail-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 24px; border-bottom: 1px solid var(--arq-border);
  position: sticky; top: 0; background: var(--arq-bg-primary); z-index: 1;
}
.mon-detail-header h3 {
  font-family: var(--arq-font-display); font-size: 0.8rem;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--arq-cyan);
}
.mon-detail-body { padding: 20px 24px; }
.mon-detail-section { margin-bottom: 24px; }
.mon-detail-section h4 {
  font-family: var(--arq-font-display); font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.15em;
  color: var(--arq-text-secondary); margin-bottom: 10px;
  padding-bottom: 6px; border-bottom: 1px solid var(--arq-border);
  font-weight: 700;
}
.mon-dl { display: grid; grid-template-columns: 130px 1fr; gap: 6px 12px; }
.mon-dl dt { color: var(--arq-text-secondary); font-size: 0.85rem; font-weight: 600; }
.mon-dl dd { font-size: 0.85rem; word-break: break-all; font-weight: 500; }
.mon-dd-val { white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
.mon-event {
  padding: 8px 12px; background: var(--arq-bg-card);
  border: 1px solid var(--arq-border); border-radius: 4px;
  margin-bottom: 8px; font-size: 0.85rem; font-weight: 500;
}
.mon-event strong { color: var(--arq-text-primary); }

/* ---- Filters ---- */
.mon-filters {
  padding: 12px 0; margin-bottom: 8px;
  border-bottom: 1px solid var(--arq-border);
}
.mon-filter-row {
  display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap;
}
.mon-filter-group { display: flex; flex-direction: column; gap: 4px; }
.mon-filter-label {
  font-family: var(--arq-font-display); font-size: 0.65rem;
  text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--arq-text-secondary); font-weight: 700;
}
.mon-input-sm { width: 200px; font-size: 0.85rem; padding: 6px 10px; font-weight: 500; }
.mon-input-xs {
  width: 75px; font-size: 0.8rem; padding: 5px 8px;
  background: var(--arq-bg-primary); color: var(--arq-text-primary);
  border: 1px solid var(--arq-border); border-radius: 4px;
  font-family: var(--arq-font-mono); outline: none;
}
.mon-input-xs:focus { border-color: var(--arq-cyan); }
.mon-filter-pills { display: flex; gap: 4px; }
.mon-pill {
  padding: 5px 12px; border-radius: 3px; cursor: pointer;
  font-family: var(--arq-font-display); font-size: 0.65rem;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
  background: var(--arq-bg-primary); color: var(--arq-text-secondary);
  border: 1px solid var(--arq-border); transition: all 0.15s;
}
.mon-pill:hover { border-color: var(--arq-border-bright); color: var(--arq-text-secondary); }
.mon-pill.active { color: #050510; }
.mon-pill-llm.active { background: #58a6ff; border-color: #58a6ff; }
.mon-pill-nerve.active { background: var(--arq-teal); border-color: var(--arq-teal); }
.mon-pill-brain.active { background: #d29922; border-color: #d29922; }
.mon-pill-dreamstate.active { background: #bc8fff; border-color: #bc8fff; }
.mon-pill-synthesis.active { background: #db6d28; border-color: #db6d28; }
.mon-pill-other.active { background: var(--arq-text-dim); border-color: var(--arq-text-dim); }
.mon-pill-err.active { background: #f85149; border-color: #f85149; }
.mon-pill-ok.active { background: var(--arq-teal); border-color: var(--arq-teal); }
.mon-filter-range { display: flex; align-items: center; gap: 6px; }
.mon-filter-dash { color: var(--arq-text-dim); }
.mon-filter-actions { flex-direction: row; align-items: center; gap: 10px; margin-left: auto; }
.mon-btn-xs { padding: 5px 12px; font-size: 0.65rem; }
.mon-filter-count {
  font-family: var(--arq-font-mono); font-size: 0.8rem;
  color: var(--arq-text-secondary); font-weight: 500;
}

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .mon-kpis { grid-template-columns: repeat(2, 1fr); }
  .mon-connect { flex-wrap: wrap; }
  .mon-wf-col-name { width: 200px; min-width: 200px; }
}
@media (max-width: 600px) {
  .mon-kpis { grid-template-columns: 1fr; }
}
</style>
