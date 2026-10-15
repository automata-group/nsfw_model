package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/recover"
	ort "github.com/yalue/onnxruntime_go"
	"go.uber.org/zap"
)

// Classification categories and ImageNet normalization constants
var (
	Categories   = []string{"drawings", "hentai", "neutral", "porn", "sexy"}
	ImageNetMean = [3]float32{0.485, 0.456, 0.406}
	ImageNetStd  = [3]float32{0.229, 0.224, 0.225}
)

type ModerationResponse struct {
	IsNSFW          bool               `json:"is_nsfw"`
	Category        string             `json:"category"`
	Confidence      float32            `json:"confidence"`
	LatencyMs       float64            `json:"latency_ms"`
	Details         map[string]float32 `json:"details"`
	AgentAction     string             `json:"agent_action"`
	AgentConfidence float32            `json:"agent_confidence"`
	AgentReasoning  string             `json:"agent_reasoning"`
	AgentRules      []string           `json:"agent_rules"`
	AgentOverride   bool               `json:"agent_override"`
}

type ModelEngine struct {
	session *ort.AdvancedSession
	mutex   sync.Mutex
	logger  *zap.Logger
}

var bufferPool = sync.Pool{
	New: func() interface{} {
		buf := make([]float32, 1*3*128*128)
		return &buf
	},
}

func NewModelEngine(modelPath string, logger *zap.Logger) (*ModelEngine, error) {
	ort.SetSharedLibraryPath(getORTSharedLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		logger.Warn("ORT environment initialization note", zap.Error(err))
	}

	inputNames := []string{"inputs"}
	outputNames := []string{"logits"}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil,
		nil,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed creating ONNX session: %w", err)
	}

	logger.Info("✅ ONNX Runtime Engine successfully initialized", zap.String("model", modelPath))
	return &ModelEngine{
		session: session,
		logger:  logger,
	}, nil
}

func getORTSharedLibPath() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	if os.PathSeparator == '\\' {
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

func (m *ModelEngine) PreprocessImage(img image.Image) *[]float32 {
	bufPtr := bufferPool.Get().(*[]float32)
	tensorData := *bufPtr

	const targetW = 128
	const targetH = 128
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	scaleX := float32(srcW) / float32(targetW)
	scaleY := float32(srcH) / float32(targetH)

	planeSize := targetW * targetH

	for y := 0; y < targetH; y++ {
		srcY := int(float32(y) * scaleY)
		for x := 0; x < targetW; x++ {
			srcX := int(float32(x) * scaleX)
			r, g, b, _ := img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY).RGBA()

			rf := float32(r>>8) / 255.0
			gf := float32(g>>8) / 255.0
			bf := float32(b>>8) / 255.0

			normR := (rf - ImageNetMean[0]) / ImageNetStd[0]
			normG := (gf - ImageNetMean[1]) / ImageNetStd[1]
			normB := (bf - ImageNetMean[2]) / ImageNetStd[2]

			idx := y*targetW + x
			tensorData[idx] = normR
			tensorData[planeSize+idx] = normG
			tensorData[2*planeSize+idx] = normB
		}
	}

	return bufPtr
}

func Softmax(logits []float32) []float32 {
	var maxLogit float32 = -math.MaxFloat32
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sumExp float64 = 0.0
	exps := make([]float64, len(logits))
	for i, v := range logits {
		e := math.Exp(float64(v - maxLogit))
		exps[i] = e
		sumExp += e
	}

	probs := make([]float32, len(logits))
	for i := range probs {
		probs[i] = float32(exps[i] / sumExp)
	}
	return probs
}

func (m *ModelEngine) Predict(inputTensorData []float32) (map[string]float32, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	inputShape := ort.NewShape(1, 3, 128, 128)
	inputTensor, err := ort.NewTensor(inputShape, inputTensorData)
	if err != nil {
		return nil, fmt.Errorf("error creating input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, int64(len(Categories)))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("error creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = m.session.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("error executing ONNX session run: %w", err)
	}

	rawLogits := outputTensor.GetData()
	probabilities := Softmax(rawLogits)

	results := make(map[string]float32, len(Categories))
	for i, cat := range Categories {
		results[cat] = probabilities[i]
	}

	return results, nil
}

type DeductiveAgent struct {
	SafeGraphicClasses map[string]bool
	NSFWClasses        map[string]bool
}

func NewDeductiveAgent() *DeductiveAgent {
	return &DeductiveAgent{
		SafeGraphicClasses: map[string]bool{"drawings": true, "neutral": true},
		NSFWClasses:        map[string]bool{"porn": true, "hentai": true, "sexy": true},
	}
}

func (a *DeductiveAgent) Decide(probs map[string]float32) (action string, conf float32, isNSFW bool, reasoning string, rules []string, override bool) {
	bestCat := "neutral"
	var maxProb float32 = -1.0
	for cat, p := range probs {
		if p > maxProb {
			maxProb = p
			bestCat = cat
		}
	}

	// Rule 1: High Confidence Hentai / 2D Anime NSFW Guard
	if bestCat == "hentai" && maxProb >= 0.50 {
		return "hentai", maxProb, true,
			"High-confidence 2D Hentai artwork verified.",
			[]string{"CONFIRMED_HENTAI_RESTRICTION"}, false
	}

	// Rule 2: Explicit Adult (Porn) Threshold
	if bestCat == "porn" && maxProb >= 0.40 {
		return bestCat, maxProb, true,
			fmt.Sprintf("Confirmed explicit adult content (%s: %.2f)", bestCat, maxProb),
			[]string{"EXPLICIT_ADULT_THRESHOLD"}, false
	}

	// Rule 2.5: Weak Hentai Hallucination Protection for Game Assets / Armor
	// Demote to drawings ONLY if explicit adult mass (hentai + porn) < 0.42 and porn < 0.10 and safe mass >= 0.42
	if bestCat == "hentai" && maxProb < 0.50 {
		explicitAdult := probs["hentai"] + probs["porn"]
		safeMass := probs["drawings"] + probs["neutral"]
		margin := probs["hentai"] - probs["drawings"]
		if probs["neutral"] > probs["drawings"] {
			margin = probs["hentai"] - probs["neutral"]
		}
		if explicitAdult < 0.42 && probs["porn"] < 0.10 && safeMass >= 0.42 && margin < 0.06 {
			return "drawings", maxProb, false,
				fmt.Sprintf("Weak hentai artifact (%.2f) on non-erotic asset demoted to safe drawings.", maxProb),
				[]string{"FALSE_HENTAI_DEMOTED_TO_DRAWING"}, true
		}
	}

	// Rule 3: Suggestive & Swimwear Protection
	if bestCat == "sexy" && maxProb >= 0.45 {
		return "sexy", maxProb, true,
			"Suggestive lingerie or exposed swimwear detected.",
			[]string{"SUGGESTIVE_SWIMWEAR_RESTRICTION"}, false
	}

	// Rule 4: Shield for Cartoon Drawings & Neutral Illustrations
	if bestCat == "drawings" || bestCat == "neutral" {
		return bestCat, maxProb, false,
			fmt.Sprintf("Safe graphical artwork verified (%s: %.2f)", bestCat, maxProb),
			[]string{"SAFE_GRAPHIC_SHIELD_ENFORCED"}, false
	}

	// Standard fallback
	isAdult := a.NSFWClasses[bestCat]
	return bestCat, maxProb, isAdult,
		fmt.Sprintf("Standard verification passed for %s.", bestCat),
		[]string{"TRUST_VISION_PREDICTION"}, false
}

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	modelPath := "nsfw_lcnet050.onnx"
	if p := os.Getenv("MODEL_PATH"); p != "" {
		modelPath = p
	}

	engine, err := NewModelEngine(modelPath, logger)
	if err != nil {
		logger.Fatal("Failed initializing ML Engine", zap.Error(err))
	}

	agent := NewDeductiveAgent()

	app := fiber.New(fiber.Config{
		BodyLimit:    20 * 1024 * 1024,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	})

	app.Use(recover.New())
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "Origin, Content-Type, Accept",
	}))
	app.Use(limiter.New(limiter.Config{
		Max:        1000,
		Expiration: 1 * time.Minute,
	}))

	app.Get("/healthz", func(c *fiber.Ctx) error {
		return c.Status(fiber.StatusOK).JSON(fiber.Map{
			"status": "healthy",
			"time":   time.Now().UTC().Format(time.RFC3339),
			"model":  modelPath,
		})
	})

	app.Post("/api/v1/moderate", func(c *fiber.Ctx) error {
		start := time.Now()

		fileHeader, err := c.FormFile("image")
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Missing multipart form file with key 'image'",
			})
		}

		file, err := fileHeader.Open()
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed opening uploaded file",
			})
		}
		defer file.Close()

		img, _, err := image.Decode(file)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Unsupported or corrupted image format",
			})
		}

		tensorPtr := engine.PreprocessImage(img)
		probabilities, err := engine.Predict(*tensorPtr)
		bufferPool.Put(tensorPtr)

		if err != nil {
			logger.Error("Inference error", zap.Error(err))
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Inference computation failed",
			})
		}

		action, conf, isNSFW, reasoning, rules, override := agent.Decide(probabilities)
		latency := float64(time.Since(start).Microseconds()) / 1000.0

		return c.Status(fiber.StatusOK).JSON(ModerationResponse{
			IsNSFW:          isNSFW,
			Category:        action,
			Confidence:      conf,
			LatencyMs:       latency,
			Details:         probabilities,
			AgentAction:     action,
			AgentConfidence: conf,
			AgentReasoning:  reasoning,
			AgentRules:      rules,
			AgentOverride:   override,
		})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	go func() {
		logger.Info("⚡ Golang NSFW Moderation Microservice started", zap.String("port", port))
		if err := app.Listen(":" + port); err != nil {
			logger.Info("Server stopped", zap.Error(err))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	<-quit

	logger.Info("Gracefully shutting down microservice...")
	_ = app.ShutdownWithTimeout(5 * time.Second)
	logger.Info("Server exited cleanly.")
}
