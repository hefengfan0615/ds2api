package client

import (
	"compress/gzip"
	"io"
	"net/http"
	"strings"

	"github.com/andybalholm/brotli"
)

func readResponseBody(resp *http.Response) ([]byte, error) {
	encoding := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Encoding")))
	var reader io.Reader = resp.Body
	switch encoding {
	case "gzip":
		gz, err := gzip.NewReader(resp.Body)
		if err != nil {
			return nil, err
		}
		defer func() { _ = gz.Close() }()
		reader = gz
	case "br":
		reader = brotli.NewReader(resp.Body)
	}
	return io.ReadAll(reader)
}

func preview(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 160 {
		return s[:160]
	}
	return s
}

func (c *Client) jsonHeaders(headers map[string]string) map[string]string {
	out := cloneStringMap(headers)
	out["Content-Type"] = "application/json"
	// 添加真实浏览器的请求头以防止被识别为爬虫
	out["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
	out["Accept"] = "application/json, text/plain, */*"
	out["Accept-Language"] = "en-US,en;q=0.9"
	out["Accept-Encoding"] = "gzip, deflate, br"
	out["Origin"] = "https://chat.deepseek.com"
	out["Referer"] = "https://chat.deepseek.com/"
	out["Sec-Ch-Ua"] = "\"Google Chrome\";v=\"125\", \"Chromium\";v=\"125\", \"Not.A/Brand\";v=\"24\""
	out["Sec-Ch-Ua-Mobile"] = "?0"
	out["Sec-Ch-Ua-Platform"] = "\"Windows\""
	out["Sec-Fetch-Dest"] = "empty"
	out["Sec-Fetch-Mode"] = "cors"
	out["Sec-Fetch-Site"] = "same-origin"
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
