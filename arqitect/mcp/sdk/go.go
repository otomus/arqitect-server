/*
SDK template for Go tool contributors.

Copy this file as your tool's entry point (main.go) and implement the handle() function.

Protocol: reads JSON-RPC from stdin, writes JSON-RPC to stdout.

Usage in tool.json:

	{"runtime": "go", "entry": "main.go"}
*/
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

type request struct {
	ID     string                 `json:"id"`
	Method string                 `json:"method"`
	Params map[string]interface{} `json:"params"`
}

type response struct {
	ID     string `json:"id"`
	Result string `json:"result,omitempty"`
	Error  string `json:"error,omitempty"`
}

// handle implements your tool logic. Replace this with your implementation.
func handle(params map[string]interface{}) (string, error) {
	return "", fmt.Errorf("replace this with your tool logic")
}

func main() {
	enc := json.NewEncoder(os.Stdout)
	enc.Encode(map[string]bool{"ready": true})

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var req request
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			continue
		}
		result, err := handle(req.Params)
		resp := response{ID: req.ID}
		if err != nil {
			resp.Error = err.Error()
		} else {
			resp.Result = result
		}
		enc.Encode(resp)
	}
}
