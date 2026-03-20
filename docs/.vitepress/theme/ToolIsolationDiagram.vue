<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const step = ref(-1)
const isVisible = ref(false)
let interval: ReturnType<typeof setInterval> | undefined
let timeout: ReturnType<typeof setTimeout> | undefined

const TOTAL_STEPS = 8
const STEP_DURATION = 2000

const steps = [
  { label: 'Nerve needs to call a tool' },
  { label: 'Nerve sends JSON-RPC request via stdin' },
  { label: 'MCP Server receives the request' },
  { label: 'Server spawns isolated subprocess' },
  { label: 'Tool subprocess executes in sandbox' },
  { label: 'Tool returns JSON-RPC response via stdout' },
  { label: 'MCP Server routes response back to nerve' },
  { label: 'Usage tracked + event published to Redis' },
]

function startAnimation() {
  step.value = 0
  interval = setInterval(() => {
    if (step.value < TOTAL_STEPS - 1) {
      step.value++
    } else {
      clearInterval(interval)
      timeout = setTimeout(() => startAnimation(), 3000)
    }
  }, STEP_DURATION)
}

onMounted(() => {
  const observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting && !isVisible.value) {
        isVisible.value = true
        startAnimation()
        observer.disconnect()
      }
    },
    { threshold: 0.3 }
  )
  if (containerRef.value) observer.observe(containerRef.value)
  onUnmounted(() => {
    observer.disconnect()
    clearInterval(interval)
    clearTimeout(timeout)
  })
})
</script>

<template>
  <div class="flow-wrapper" ref="containerRef">
    <!-- Step indicator -->
    <div class="step-label" :class="{ visible: isVisible }">
      <span class="step-text" :key="step">{{ steps[step]?.label ?? '' }}</span>
    </div>

    <svg
      class="flow-diagram"
      :class="{ visible: isVisible }"
      viewBox="0 0 1000 420"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="ti-gl-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00d4ff" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="ti-gl-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00ff88" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="ti-gl-yellow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#facc15" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="ti-gl-orange" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff6a00" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <marker id="ti-arr-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff" />
        </marker>
        <marker id="ti-arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88" />
        </marker>
        <marker id="ti-arr-yellow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#facc15" />
        </marker>
        <marker id="ti-arr-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6a00" />
        </marker>
      </defs>

      <!-- ===== NERVE NODE (left) ===== -->
      <g :class="['node', { active: step >= 0 }]">
        <rect
          x="40" y="130" width="180" height="60" rx="6"
          fill="rgba(0,255,136,0.06)" stroke="#00ff88" stroke-width="2"
          :filter="step === 0 || step === 6 ? 'url(#ti-gl-green)' : ''"
        />
        <text x="130" y="155" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="14" font-weight="700" letter-spacing="0.12em">NERVE</text>
        <text x="130" y="177" text-anchor="middle" fill="rgba(0,255,136,0.6)" font-family="'Share Tech Mono',monospace" font-size="13">
          {{ step >= 6 ? 'result received' : 'calendar-nerve' }}
        </text>
      </g>

      <!-- ===== ARROW: Nerve → MCP Server (step 1) ===== -->
      <line
        x1="220" y1="160" x2="370" y2="160"
        :class="['arrow', { active: step >= 1 }]"
        stroke="#00ff88" stroke-width="2" marker-end="url(#ti-arr-green)"
      />
      <circle v-if="step === 1" r="5" fill="#00ff88" filter="url(#ti-gl-green)" class="particle">
        <animateMotion dur="0.8s" repeatCount="indefinite" path="M220,160 L370,160" />
      </circle>
      <!-- stdin label -->
      <text
        x="295" y="148"
        text-anchor="middle" fill="rgba(0,255,136,0.5)"
        font-family="'Share Tech Mono',monospace" font-size="12"
        :class="['node', { active: step >= 1 }]"
      >stdin</text>

      <!-- ===== MCP SERVER NODE (center) ===== -->
      <g :class="['node', { active: step >= 2 }]">
        <rect
          x="380" y="115" width="200" height="90" rx="8"
          fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="2.5"
          :filter="step >= 2 && step <= 3 || step >= 5 ? 'url(#ti-gl-cyan)' : ''"
        />
        <rect
          x="390" y="125" width="180" height="70" rx="4"
          fill="none" stroke="#00d4ff" stroke-width="0.6" stroke-opacity="0.25" stroke-dasharray="4 5"
          :class="{ spinning: step >= 2 }"
        />
        <text x="480" y="155" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="15" font-weight="900" letter-spacing="0.18em">MCP SERVER</text>
        <text x="480" y="178" text-anchor="middle" fill="rgba(0,212,255,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step === 3 ? 'spawning...' : (step >= 5 ? 'routing response' : 'json-rpc router') }}
        </text>
      </g>

      <!-- ===== ARROW: MCP Server → Tool subprocess (step 3) ===== -->
      <line
        x1="580" y1="160" x2="680" y2="160"
        :class="['arrow', { active: step >= 3 }]"
        stroke="#00d4ff" stroke-width="2" marker-end="url(#ti-arr-cyan)"
      />
      <circle v-if="step === 3" r="5" fill="#00d4ff" filter="url(#ti-gl-cyan)" class="particle">
        <animateMotion dur="0.6s" repeatCount="indefinite" path="M580,160 L680,160" />
      </circle>
      <!-- spawn label -->
      <text
        x="630" y="148"
        text-anchor="middle" fill="rgba(0,212,255,0.5)"
        font-family="'Share Tech Mono',monospace" font-size="12"
        :class="['node', { active: step >= 3 }]"
      >spawn</text>

      <!-- ===== TOOL SUBPROCESS NODE (right) ===== -->
      <g :class="['node', { active: step >= 3 }]">
        <!-- Sandbox border -->
        <rect
          x="688" y="100" width="270" height="120" rx="10"
          fill="none" stroke="#facc15" stroke-width="1" stroke-opacity="0.3" stroke-dasharray="6 4"
        />
        <text x="823" y="118" text-anchor="middle" fill="rgba(250,204,21,0.4)" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.15em">ISOLATED SUBPROCESS</text>

        <!-- Inner tool box -->
        <rect
          x="708" y="128" width="230" height="72" rx="5"
          fill="rgba(250,204,21,0.06)" stroke="#facc15" stroke-width="1.5"
          :filter="step === 4 ? 'url(#ti-gl-yellow)' : ''"
        />
        <text x="823" y="155" text-anchor="middle" fill="#facc15" font-family="'Orbitron',sans-serif" font-size="13" font-weight="700" letter-spacing="0.1em">TOOL</text>
        <text x="823" y="175" text-anchor="middle" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="13">
          {{ step === 4 ? 'executing...' : 'weather.py' }}
        </text>
        <text x="823" y="192" text-anchor="middle" fill="rgba(250,204,21,0.35)" font-family="'Share Tech Mono',monospace" font-size="11">
          pid: {{ step >= 3 ? '48291' : '—' }}
        </text>
      </g>

      <!-- ===== ARROW: Tool → MCP Server (step 5, return) ===== -->
      <line
        x1="688" y1="185" x2="580" y2="185"
        :class="['arrow', 'return-arrow', { active: step >= 5 }]"
        stroke="#facc15" stroke-width="2" marker-end="url(#ti-arr-yellow)"
      />
      <circle v-if="step === 5" r="5" fill="#facc15" filter="url(#ti-gl-yellow)" class="particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M688,185 L580,185" />
      </circle>
      <!-- stdout label -->
      <text
        x="634" y="198"
        text-anchor="middle" fill="rgba(250,204,21,0.5)"
        font-family="'Share Tech Mono',monospace" font-size="12"
        :class="['node', { active: step >= 5 }]"
      >stdout</text>

      <!-- ===== ARROW: MCP Server → Nerve (step 6, return) ===== -->
      <line
        x1="380" y1="185" x2="220" y2="185"
        :class="['arrow', 'return-arrow', { active: step >= 6 }]"
        stroke="#00d4ff" stroke-width="2" marker-end="url(#ti-arr-cyan)"
      />
      <circle v-if="step === 6" r="5" fill="#00d4ff" filter="url(#ti-gl-cyan)" class="particle">
        <animateMotion dur="0.8s" repeatCount="indefinite" path="M380,185 L220,185" />
      </circle>

      <!-- ===== ARROW: MCP Server → Monitoring (step 7) ===== -->
      <line
        x1="440" y1="205" x2="340" y2="320"
        :class="['arrow', { active: step >= 7 }]"
        stroke="#ff6a00" stroke-width="1.5" marker-end="url(#ti-arr-orange)"
        stroke-dasharray="5 4"
      />
      <circle v-if="step === 7" r="4" fill="#ff6a00" filter="url(#ti-gl-orange)" class="particle">
        <animateMotion dur="0.9s" repeatCount="indefinite" path="M440,205 L340,320" />
      </circle>

      <!-- ===== MONITORING NODE (bottom-left) ===== -->
      <g :class="['node', { active: step >= 7 }]">
        <rect
          x="240" y="325" width="200" height="60" rx="5"
          fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5"
          :filter="step === 7 ? 'url(#ti-gl-orange)' : ''"
        />
        <text x="340" y="350" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.1em">MONITORING</text>
        <text x="340" y="372" text-anchor="middle" fill="rgba(255,106,0,0.55)" font-family="'Share Tech Mono',monospace" font-size="12">best-effort tracking</text>
      </g>

      <!-- ===== ARROW: MCP Server → Redis (step 7) ===== -->
      <line
        x1="520" y1="205" x2="620" y2="320"
        :class="['arrow', { active: step >= 7 }]"
        stroke="#ff6a00" stroke-width="1.5" marker-end="url(#ti-arr-orange)"
        stroke-dasharray="5 4"
      />
      <circle v-if="step === 7" r="4" fill="#ff6a00" filter="url(#ti-gl-orange)" class="particle">
        <animateMotion dur="1.0s" repeatCount="indefinite" path="M520,205 L620,320" />
      </circle>

      <!-- ===== REDIS NODE (bottom-right) ===== -->
      <g :class="['node', { active: step >= 7 }]">
        <rect
          x="520" y="325" width="200" height="60" rx="5"
          fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5"
          :filter="step === 7 ? 'url(#ti-gl-orange)' : ''"
        />
        <text x="620" y="350" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.1em">REDIS</text>
        <text x="620" y="372" text-anchor="middle" fill="rgba(255,106,0,0.55)" font-family="'Share Tech Mono',monospace" font-size="12">event published</text>
      </g>

      <!-- ===== Step progress dots ===== -->
      <g transform="translate(412, 410)">
        <circle
          v-for="(s, i) in steps" :key="'dot-' + i"
          :cx="i * 22" cy="0" r="4"
          :fill="i === step ? '#00d4ff' : (i < step ? 'rgba(0,212,255,0.4)' : 'rgba(0,212,255,0.1)')"
          :class="{ 'dot-active': i === step }"
        />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.flow-wrapper {
  max-width: 1000px;
  margin: 2rem auto 0;
  padding: 0 1rem;
}

.step-label {
  text-align: center;
  min-height: 36px;
  margin-bottom: 0.5rem;
  opacity: 0;
  transition: opacity 0.6s;
}

.step-label.visible {
  opacity: 1;
}

.step-text {
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.1rem;
  color: rgba(0, 212, 255, 0.8);
  letter-spacing: 0.05em;
  animation: ti-text-appear 0.4s ease forwards;
}

@keyframes ti-text-appear {
  0% { opacity: 0; transform: translateY(6px); filter: blur(4px); }
  100% { opacity: 1; transform: translateY(0); filter: blur(0); }
}

.flow-diagram {
  width: 100%;
  height: auto;
  opacity: 0;
  transition: opacity 0.8s ease;
}

.flow-diagram.visible {
  opacity: 1;
}

.node {
  opacity: 0.15;
  transition: opacity 0.5s ease;
}

.node.active {
  opacity: 1;
}

.arrow {
  opacity: 0;
  stroke-dasharray: 400;
  stroke-dashoffset: 400;
  transition: opacity 0.4s ease, stroke-dashoffset 0.6s ease;
}

.arrow.active {
  opacity: 0.6;
  stroke-dashoffset: 0;
}

.return-arrow.active {
  stroke-dasharray: 8 5;
  stroke-dashoffset: 0;
  animation: ti-dash-flow 0.8s linear infinite;
}

@keyframes ti-dash-flow {
  to { stroke-dashoffset: -26; }
}

.particle {
  opacity: 0.9;
}

.spinning {
  transform-origin: 480px 160px;
  animation: ti-spin 8s linear infinite;
}

@keyframes ti-spin {
  to { transform: rotate(360deg); }
}

.dot-active {
  animation: ti-dot-pulse 1s ease-in-out infinite;
}

@keyframes ti-dot-pulse {
  0%, 100% { r: 4; }
  50% { r: 6; }
}

@media (max-width: 640px) {
  .flow-wrapper {
    margin: 1rem auto 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .flow-diagram {
    min-width: 600px;
  }
  .step-text {
    font-size: 0.85rem;
  }
}
</style>
