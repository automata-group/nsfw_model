class ModerationResult {
  final bool isNsfw;
  final String category;
  final double confidence;
  final double latencyMs;
  final Map<String, double> details;
  final String agentAction;
  final double agentConfidence;
  final String agentReasoning;
  final List<String> agentRules;
  final bool agentOverride;

  const ModerationResult({
    required this.isNsfw,
    required this.category,
    required this.confidence,
    required this.latencyMs,
    required this.details,
    required this.agentAction,
    required this.agentConfidence,
    required this.agentReasoning,
    required this.agentRules,
    required this.agentOverride,
  });

  factory ModerationResult.fromJson(Map<String, dynamic> json) {
    final rawDetails = json['details'] as Map<String, dynamic>? ?? {};
    final parsedDetails = rawDetails.map(
      (k, v) => MapEntry(k, (v as num).toDouble()),
    );

    return ModerationResult(
      isNsfw: json['is_nsfw'] as bool? ?? false,
      category: json['category'] as String? ?? 'unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      latencyMs: (json['latency_ms'] as num?)?.toDouble() ?? 0.0,
      details: parsedDetails,
      agentAction: json['agent_action'] as String? ?? 'unknown',
      agentConfidence: (json['agent_confidence'] as num?)?.toDouble() ?? 0.0,
      agentReasoning: json['agent_reasoning'] as String? ?? '',
      agentRules: (json['agent_rules'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      agentOverride: json['agent_override'] as bool? ?? false,
    );
  }
}
