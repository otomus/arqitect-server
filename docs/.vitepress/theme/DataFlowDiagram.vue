<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const step = ref(-1)
const isVisible = ref(false)
let interval: ReturnType<typeof setInterval> | undefined
let timeout: ReturnType<typeof setTimeout> | undefined

const TOTAL_STEPS = 10
const STEP_DURATION = 2000

const steps = [
  { label: 'User sends a task' },
  { label: 'Brain.think() receives the task' },
  { label: 'Intent classification — workflow or direct?' },
  { label: 'LLM generates a typed action decision' },
  { label: 'Normalize \u2192 Validate \u2192 Dispatch' },
  { label: 'Handler executes the decision' },
  { label: 'Nerve subprocess runs with senses + tools' },
  { label: 'Result recorded as an episode' },
  { label: 'Communication sense rewrites the response' },
  { label: 'Response returned to the user' },
]

function startAnimation(): void {
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
    { threshold: 0.2 }
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
  <div class="df-wrapper" ref="containerRef">
    <div class="df-step-label" :class="{ visible: isVisible }">
      <span class="df-step-text" :key="step">{{ steps[step]?.label ?? '' }}</span>
    </div>

    <svg
      class="df-diagram"
      :class="{ visible: isVisible }"
      viewBox="0 0 1000 520"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="df-gl-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00d4ff" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="df-gl-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00ff88" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="df-gl-orange" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff6a00" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="df-gl-magenta" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff00aa" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="df-gl-yellow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#facc15" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="df-gl-purple" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feFlood flood-color="#a855f7" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <marker id="df-arr-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff" />
        </marker>
        <marker id="df-arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88" />
        </marker>
        <marker id="df-arr-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6a00" />
        </marker>
        <marker id="df-arr-magenta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff00aa" />
        </marker>
        <marker id="df-arr-yellow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#facc15" />
        </marker>
        <marker id="df-arr-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7" />
        </marker>
      </defs>

      <!-- ===== ROW 1: User → Brain → Intent → Decision ===== -->

      <!-- USER NODE -->
      <g :class="['df-node', { active: step >= 0 }]">
        <circle cx="60" cy="90" r="40" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="2" :filter="step === 0 || step === 9 ? 'url(#df-gl-cyan)' : ''" />
        <text x="60" y="82" text-anchor="middle" fill="#00d4ff" font-size="24">&#x1F464;</text>
        <text x="60" y="104" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.12em">USER</text>
      </g>

      <!-- Arrow: User → Brain -->
      <line x1="104" y1="90" x2="175" y2="90" :class="['df-arrow', { active: step >= 0 }]" stroke="#00d4ff" stroke-width="2" marker-end="url(#df-arr-cyan)" />
      <circle v-if="step === 0" r="4" fill="#00d4ff" filter="url(#df-gl-cyan)" class="df-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M104,90 L175,90" />
      </circle>

      <!-- BRAIN NODE -->
      <g :class="['df-node', { active: step >= 1 }]">
        <circle cx="230" cy="90" r="50" fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="2.5" :filter="step === 1 ? 'url(#df-gl-cyan)' : ''" />
        <circle cx="230" cy="90" r="38" fill="none" stroke="#00d4ff" stroke-width="0.8" stroke-opacity="0.3" stroke-dasharray="4 5" :class="{ 'df-spinning': step >= 1 }" />
        <text x="230" y="86" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="15" font-weight="900" letter-spacing="0.18em">BRAIN</text>
        <text x="230" y="104" text-anchor="middle" fill="rgba(0,212,255,0.5)" font-family="'Share Tech Mono',monospace" font-size="11">.think()</text>
      </g>

      <!-- Arrow: Brain → Intent -->
      <line x1="284" y1="90" x2="355" y2="90" :class="['df-arrow', { active: step >= 2 }]" stroke="#00d4ff" stroke-width="2" marker-end="url(#df-arr-cyan)" />
      <circle v-if="step === 2" r="4" fill="#00d4ff" filter="url(#df-gl-cyan)" class="df-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M284,90 L355,90" />
      </circle>

      <!-- INTENT CLASSIFICATION NODE (diamond shape via rotated rect) -->
      <g :class="['df-node', { active: step >= 2 }]">
        <rect x="360" y="62" width="120" height="56" rx="5" fill="rgba(0,255,136,0.05)" stroke="#00ff88" stroke-width="1.5" :filter="step === 2 ? 'url(#df-gl-green)' : ''" />
        <text x="420" y="84" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.08em">INTENT</text>
        <text x="420" y="104" text-anchor="middle" fill="rgba(0,255,136,0.6)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step >= 2 ? 'workflow | direct' : '' }}
        </text>
      </g>

      <!-- Arrow: Intent → Decision -->
      <line x1="484" y1="90" x2="555" y2="90" :class="['df-arrow', { active: step >= 3 }]" stroke="#00ff88" stroke-width="2" marker-end="url(#df-arr-green)" />
      <circle v-if="step === 3" r="4" fill="#00ff88" filter="url(#df-gl-green)" class="df-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M484,90 L555,90" />
      </circle>

      <!-- LLM DECISION NODE -->
      <g :class="['df-node', { active: step >= 3 }]">
        <rect x="560" y="44" width="180" height="92" rx="6" fill="rgba(250,204,21,0.05)" stroke="#facc15" stroke-width="1.5" :filter="step === 3 ? 'url(#df-gl-yellow)' : ''" />
        <text x="650" y="66" text-anchor="middle" fill="#facc15" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.1em">LLM DECISION</text>
        <text x="610" y="84" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'invoke' : '' }}
        </text>
        <text x="690" y="84" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'synthesize' : '' }}
        </text>
        <text x="610" y="100" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'chain' : '' }}
        </text>
        <text x="690" y="100" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'clarify' : '' }}
        </text>
        <text x="610" y="116" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'sense' : '' }}
        </text>
        <text x="690" y="116" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'respond' : '' }}
        </text>
        <text x="610" y="132" text-anchor="start" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step >= 3 ? 'feedback' : '' }}
        </text>
      </g>

      <!-- ===== ROW 2: Normalize → Validate → Dispatch pipeline ===== -->

      <!-- Arrow: Decision ↓ to pipeline -->
      <line x1="650" y1="140" x2="650" y2="185" :class="['df-arrow', { active: step >= 4 }]" stroke="#facc15" stroke-width="2" marker-end="url(#df-arr-yellow)" />
      <circle v-if="step === 4" r="4" fill="#facc15" filter="url(#df-gl-yellow)" class="df-particle">
        <animateMotion dur="0.5s" repeatCount="indefinite" path="M650,140 L650,185" />
      </circle>

      <!-- NORMALIZE BOX -->
      <g :class="['df-node', 'df-pipeline', { active: step >= 4 }]">
        <rect x="540" y="190" width="100" height="40" rx="4" fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5" :filter="step === 4 ? 'url(#df-gl-orange)' : ''" />
        <text x="590" y="215" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">NORMALIZE</text>
      </g>

      <!-- Arrow: Normalize → Validate -->
      <line x1="644" y1="210" x2="660" y2="210" :class="['df-arrow', 'df-pipe-arrow', { active: step >= 4 }]" stroke="#ff6a00" stroke-width="1.5" marker-end="url(#df-arr-orange)" />

      <!-- VALIDATE BOX -->
      <g :class="['df-node', 'df-pipeline', { active: step >= 4 }]">
        <rect x="664" y="190" width="100" height="40" rx="4" fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5" :filter="step === 4 ? 'url(#df-gl-orange)' : ''" />
        <text x="714" y="215" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">VALIDATE</text>
      </g>

      <!-- Arrow: Validate → Dispatch -->
      <line x1="768" y1="210" x2="784" y2="210" :class="['df-arrow', 'df-pipe-arrow', { active: step >= 4 }]" stroke="#ff6a00" stroke-width="1.5" marker-end="url(#df-arr-orange)" />

      <!-- DISPATCH BOX -->
      <g :class="['df-node', 'df-pipeline', { active: step >= 4 }]">
        <rect x="788" y="190" width="100" height="40" rx="4" fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5" :filter="step === 4 ? 'url(#df-gl-orange)' : ''" />
        <text x="838" y="215" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">DISPATCH</text>
      </g>

      <!-- ===== ROW 3: Handler → Nerve ===== -->

      <!-- Arrow: Dispatch ↓ to Handler -->
      <line x1="838" y1="234" x2="838" y2="275" :class="['df-arrow', { active: step >= 5 }]" stroke="#ff6a00" stroke-width="2" marker-end="url(#df-arr-orange)" />
      <circle v-if="step === 5" r="4" fill="#ff6a00" filter="url(#df-gl-orange)" class="df-particle">
        <animateMotion dur="0.5s" repeatCount="indefinite" path="M838,234 L838,275" />
      </circle>

      <!-- HANDLER BOX -->
      <g :class="['df-node', { active: step >= 5 }]">
        <rect x="758" y="280" width="160" height="50" rx="5" fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="1.5" :filter="step === 5 ? 'url(#df-gl-orange)' : ''" />
        <text x="838" y="302" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.08em">HANDLER</text>
        <text x="838" y="320" text-anchor="middle" fill="rgba(255,106,0,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">invoke_nerve</text>
      </g>

      <!-- Arrow: Handler → Nerve -->
      <line x1="754" y1="305" x2="620" y2="305" :class="['df-arrow', { active: step >= 6 }]" stroke="#00ff88" stroke-width="2" marker-end="url(#df-arr-green)" />
      <circle v-if="step === 6" r="4" fill="#00ff88" filter="url(#df-gl-green)" class="df-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M754,305 L620,305" />
      </circle>

      <!-- NERVE SUBPROCESS NODE -->
      <g :class="['df-node', { active: step >= 6 }]">
        <rect x="460" y="272" width="156" height="66" rx="6" fill="rgba(0,255,136,0.06)" stroke="#00ff88" stroke-width="2" :filter="step === 6 ? 'url(#df-gl-green)' : ''" />
        <text x="538" y="296" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.1em">NERVE</text>
        <text x="538" y="316" text-anchor="middle" fill="rgba(0,255,136,0.5)" font-family="'Share Tech Mono',monospace" font-size="11">senses + tools</text>
        <text x="538" y="332" text-anchor="middle" fill="rgba(0,255,136,0.35)" font-family="'Share Tech Mono',monospace" font-size="10">subprocess</text>
      </g>

      <!-- ===== ROW 4: Result → Memory → Comm → User ===== -->

      <!-- Arrow: Nerve ↓ to Memory -->
      <line x1="538" y1="342" x2="538" y2="385" :class="['df-arrow', { active: step >= 7 }]" stroke="#a855f7" stroke-width="2" marker-end="url(#df-arr-purple)" />
      <circle v-if="step === 7" r="4" fill="#a855f7" filter="url(#df-gl-purple)" class="df-particle">
        <animateMotion dur="0.5s" repeatCount="indefinite" path="M538,342 L538,385" />
      </circle>

      <!-- MEMORY / EPISODE NODE -->
      <g :class="['df-node', { active: step >= 7 }]">
        <rect x="460" y="390" width="156" height="50" rx="5" fill="rgba(168,85,247,0.06)" stroke="#a855f7" stroke-width="1.5" :filter="step === 7 ? 'url(#df-gl-purple)' : ''" />
        <text x="538" y="412" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.08em">MEMORY</text>
        <text x="538" y="430" text-anchor="middle" fill="rgba(168,85,247,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">episode recorded</text>
      </g>

      <!-- Arrow: Memory → Communication -->
      <line x1="456" y1="415" x2="340" y2="415" :class="['df-arrow', { active: step >= 8 }]" stroke="#ff00aa" stroke-width="2" marker-end="url(#df-arr-magenta)" />
      <circle v-if="step === 8" r="4" fill="#ff00aa" filter="url(#df-gl-magenta)" class="df-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M456,415 L340,415" />
      </circle>

      <!-- COMMUNICATION SENSE NODE -->
      <g :class="['df-node', { active: step >= 8 }]">
        <rect x="170" y="390" width="166" height="50" rx="5" fill="rgba(255,0,170,0.06)" stroke="#ff00aa" stroke-width="1.5" :filter="step === 8 ? 'url(#df-gl-magenta)' : ''" />
        <text x="253" y="412" text-anchor="middle" fill="#ff00aa" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.08em">COMMUNICATION</text>
        <text x="253" y="430" text-anchor="middle" fill="rgba(255,0,170,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step === 8 ? 'rewriting...' : 'sense' }}
        </text>
      </g>

      <!-- Arrow: Communication → User (curves up-left) -->
      <path
        d="M 170,415 L 100,415 Q 60,415 60,375 L 60,134"
        fill="none"
        :class="['df-arrow', 'df-return', { active: step >= 9 }]"
        stroke="#00ff88" stroke-width="2" marker-end="url(#df-arr-green)"
      />
      <circle v-if="step === 9" r="4" fill="#00ff88" filter="url(#df-gl-green)" class="df-particle">
        <animateMotion dur="1.2s" repeatCount="indefinite" path="M170,415 L100,415 Q60,415 60,375 L60,134" />
      </circle>

      <!-- ===== RE-THINK LOOP (dashed, always visible once step >= 7) ===== -->
      <path
        d="M 460,310 L 340,310 Q 290,310 290,260 L 290,180 Q 290,144 240,144 L 230,144"
        fill="none"
        :class="['df-arrow', 'df-rethink', { active: step >= 7 }]"
        stroke="rgba(0,212,255,0.3)" stroke-width="1.5" stroke-dasharray="6 4"
      />
      <text x="310" y="260" fill="rgba(0,212,255,0.25)" font-family="'Share Tech Mono',monospace" font-size="10" :opacity="step >= 7 ? 1 : 0" class="df-rethink-label">re-think</text>

      <!-- ===== CIRCUIT BREAKER badge ===== -->
      <g :class="['df-node', { active: step >= 7 }]">
        <rect x="295" y="180" width="110" height="28" rx="4" fill="rgba(255,106,0,0.04)" stroke="rgba(255,106,0,0.25)" stroke-width="1" />
        <text x="350" y="199" text-anchor="middle" fill="rgba(255,106,0,0.4)" font-family="'Share Tech Mono',monospace" font-size="10">circuit breaker</text>
      </g>

      <!-- ===== Step progress dots ===== -->
      <g transform="translate(390, 490)">
        <circle
          v-for="(s, i) in steps" :key="'df-dot-' + i"
          :cx="i * 22" cy="0" r="4"
          :fill="i === step ? '#00d4ff' : (i < step ? 'rgba(0,212,255,0.4)' : 'rgba(0,212,255,0.1)')"
          :class="{ 'df-dot-active': i === step }"
        />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.df-wrapper {
  max-width: 1000px;
  margin: 2rem auto 0;
  padding: 0 1rem;
}

.df-step-label {
  text-align: center;
  min-height: 36px;
  margin-bottom: 0.5rem;
  opacity: 0;
  transition: opacity 0.6s;
}

.df-step-label.visible {
  opacity: 1;
}

.df-step-text {
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.1rem;
  color: rgba(0, 212, 255, 0.8);
  letter-spacing: 0.05em;
  animation: df-text-appear 0.4s ease forwards;
}

@keyframes df-text-appear {
  0% { opacity: 0; transform: translateY(6px); filter: blur(4px); }
  100% { opacity: 1; transform: translateY(0); filter: blur(0); }
}

.df-diagram {
  width: 100%;
  height: auto;
  opacity: 0;
  transition: opacity 0.8s ease;
}

.df-diagram.visible {
  opacity: 1;
}

.df-node {
  opacity: 0.15;
  transition: opacity 0.5s ease;
}

.df-node.active {
  opacity: 1;
}

.df-arrow {
  opacity: 0;
  stroke-dasharray: 400;
  stroke-dashoffset: 400;
  transition: opacity 0.4s ease, stroke-dashoffset 0.6s ease;
}

.df-arrow.active {
  opacity: 0.6;
  stroke-dashoffset: 0;
}

.df-pipe-arrow.active {
  opacity: 0.5;
}

.df-return.active {
  stroke-dasharray: 8 5;
  stroke-dashoffset: 0;
  animation: df-dash-flow 0.8s linear infinite;
}

.df-rethink {
  opacity: 0;
  transition: opacity 0.6s ease;
}

.df-rethink.active {
  opacity: 0.4;
  stroke-dashoffset: 0;
  animation: df-dash-flow 1.2s linear infinite;
}

.df-rethink-label {
  transition: opacity 0.6s ease;
}

@keyframes df-dash-flow {
  to { stroke-dashoffset: -26; }
}

.df-particle {
  opacity: 0.9;
}

.df-spinning {
  transform-origin: 230px 90px;
  animation: df-spin 8s linear infinite;
}

@keyframes df-spin {
  to { transform: rotate(360deg); }
}

.df-dot-active {
  animation: df-dot-pulse 1s ease-in-out infinite;
}

@keyframes df-dot-pulse {
  0%, 100% { r: 4; }
  50% { r: 6; }
}

@media (max-width: 640px) {
  .df-wrapper {
    margin: 1rem auto 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .df-diagram {
    min-width: 600px;
  }
  .df-step-text {
    font-size: 0.85rem;
  }
}
</style>
