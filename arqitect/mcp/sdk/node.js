/**
 * SDK template for Node.js tool contributors.
 *
 * Copy this file as your tool's entry point (run.js) and implement the handle() function.
 *
 * Protocol: reads JSON-RPC from stdin, writes JSON-RPC to stdout.
 *
 * Usage in tool.json:
 *     {"runtime": "node", "entry": "run.js"}
 */

const readline = require('readline');

/**
 * Implement your tool logic here.
 * @param {Object} params - The parameters passed by the caller.
 * @returns {string} Result string (use JSON.stringify for structured data).
 */
function handle(params) {
  throw new Error('Replace this with your tool logic');
}

// Stdio JSON-RPC loop — do not modify below this line.
process.stdout.write(JSON.stringify({ ready: true }) + '\n');

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let req;
  try {
    req = JSON.parse(line);
  } catch {
    return;
  }
  let resp;
  try {
    const result = handle(req.params || {});
    resp = { id: req.id, result: String(result) };
  } catch (e) {
    resp = { id: req.id, error: e.message };
  }
  process.stdout.write(JSON.stringify(resp) + '\n');
});
