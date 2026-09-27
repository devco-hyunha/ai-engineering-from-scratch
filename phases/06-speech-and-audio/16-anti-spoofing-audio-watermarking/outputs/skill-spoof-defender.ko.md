---
name: spoof-defender
description: 음성 생성 및 음성 인증 배포를 위한 탐지 모델, 워터마크, 출처 증명(provenance) 매니페스트 및 운영 플레이북을 선정합니다.
version: 1.0.0
phase: 6
lesson: 16
tags: [anti-spoofing, watermark, audioseal, asvspoof, c2pa, voice-fraud]
---

작업 부하(음성 생성 vs 음성 인증, 배포 규모, 규제 지역, 공격자 프로필)를 고려하여 다음 항목을 출력하세요:

1. **탐지(Detection, CM)**: AASIST · RawNet2 · NeXt-TDNN + WavLM · 상용 솔루션(Pindrop, Validsoft). 학습 데이터: ASVspoof 2019 / ASVspoof 5 / 도메인 특화 데이터. 목표 EER(Equal Error Rate).
2. **워터마킹(Watermarking, 생성물 출력 시)**: AudioSeal 16-bit 페이로드 인코딩 `(model_id, user_id, generation_ts)` · WaveVerify (대안) · 없음 (사유 포함). 탐지기는 출고 전 모든 출력물에 대해 CI(지속적 통합) 환경에서 실행됩니다.
3. **출처 증명(Provenance)**: 배포자의 키로 서명된 C2PA 매니페스트 · IPTC 메타데이터 · 없음 (비소비자용 오디오의 경우).
4. **음성 인증 가드(Voice-auth guards, 해당되는 경우)**: 생존 확인 챌린지(Liveness challenge, 무작위 문구 TTS 생성 + 전사), 재전송 공격 탐지(Replay attack detection, AASIST + PA 모델), 채널별 생체 인식 임계값 보정(Biometric threshold calibration).
5. **운영(Operational)**: 감사 로그 보관, 동의 증빙 자료 보관(7년 이상), 남용 탐지 신호(갑작스러운 볼륨 급증, 개체명 프롬프트), 킬 스위치(Kill-switch) 절차.

**거부 조건:**
- AudioSeal(또는 이에 상응하는 워터마크)이 없는 음성 생성 배포는 거부하십시오.
- 안티 스푸핑(Anti-spoofing) 탐지 기능이 없는 음성 생체 인증 배포는 거부하십시오.
- 음성 복제(Voice cloning) 기술로 인해 코사인 유사도(Cosine-only) 기반 인증은 매우 쉽게 우회될 수 있으므로 거부하십시오.
- 출처 증명 매니페스트에만 의존하는 배포(제거 가능함)는 거부하십시오.
- 채널 보정 스윕(Channel-calibration sweep) 없이 ASVspoof 2019로만 학습된 탐지 임계값을 실제 환경 배포에 사용하는 것은 거부하십시오.

**입력 예시:** "은행 고객 서비스 IVR. 음성 생체 인증 잠금 해제 + AI 생성 음성 에이전트. 월 1,000만 건의 통화. 미국 + EU."

**출력 예시:**
- **탐지(Detection)**: Pindrop 상용 솔루션(권장) 또는 NeXt-TDNN + WavLM 오픈 소스. ASVspoof 5 + 10만 건의 은행 특화 통화 샘플로 학습. 도메인 내 데이터 기준 목표 EER < 0.5%.
- **워터마킹(Watermarking)**: 모든 출력 TTS 발화에 AudioSeal 16-bit 페이로드 적용; 페이로드는 `bank_id` + `session_id` + `timestamp`를 인코딩함. 전송 전 탐지기가 검증함.
- **출처 증명(Provenance)**: 고객 대상 오디오 내보내기 워크플로에 C2PA 매니페스트 적용; 내부 전용 통화는 제외.
- **음성 인증(Voice-auth)**: 모든 인증 시 생존 확인 챌린지 수행(TTS 무작위 4자리 문구 생성; 사용자가 반복 + 탐지기 + 전사기). 모든 유입 인증 시도에 대해 안티 스푸핑 실행. 생체 인식 임계값은 FAR 0.1%, FRR 1%로 설정.
- **운영(Operational)**: 해당 지역 내 동의 및 감사 로그 7년 보관(EU 데이터는 EU 거주자 데이터 기준). 갑작스러운 복제 요청 볼륨이 2σ를 초과할 경우 경고; 남용 탐지 시 킬 스위치 작동.
