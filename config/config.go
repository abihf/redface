package config

import (
	"encoding/json"
	"log/slog"
	"os"
)

type Config struct {
	Device    string  `json:"device"`
	Threshold float64 `json:"threshold"`
	Timeout   int     `json:"timeout"`
	Socket    string  `json:"socket"`
	PidFile   string  `json:"pid_file"`
}

func Load() *Config {
	conf, err := loadFromFile()
	if err != nil {
		slog.Warn("Failed to load config file", "error", err)
	}
	if conf == nil {
		conf = &Config{}
	}
	if conf.Device == "" {
		panic("Device not set")
	}
	if conf.Threshold == 0 {
		conf.Threshold = 0.1
	}
	if conf.Timeout == 0 {
		conf.Timeout = 10
	}
	if conf.Socket == "" {
		conf.Socket = "/var/run/redface.sock"
	}
	if conf.PidFile == "" {
		conf.PidFile = "/var/run/redface.pid"
	}

	return conf
}

func loadFromFile() (*Config, error) {
	file, err := os.Open("/etc/redface/config.json")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	config := &Config{}
	err = json.NewDecoder(file).Decode(config)
	if err != nil {
		return nil, err
	}

	return config, nil
}
