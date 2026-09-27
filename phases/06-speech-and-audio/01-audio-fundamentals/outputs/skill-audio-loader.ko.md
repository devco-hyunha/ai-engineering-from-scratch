---
name: audio-loader
description: 원본 오디오 파일을 대상 모델의 요구 사항에 맞춰 검증하고 안전하게 리샘플링합니다.
version: 1.0.0
phase: 6
lesson: 01
tags: [audio, speech, preprocessing]
---

오디오 파일 정보(경로, 채널, 샘플 레이트, 비트 심도, 코덱)와 대상 모델 정보(필요한 샘플 레이트 및 채널 수를 가진 ASR / TTS / 분류기)가 주어지면 다음을 출력합니다:

1. 불일치 사항(Mismatches): 파일이 대상 모델의 사양과 일치하지 않는 모든 차원을 나열합니다 (샘플 레이트, 채널, 지속 시간 하한선, 클리핑 여부).
2. 리샘플링 계획(Resample plan): 소스 샘플 레이트, 대상 샘플 레이트, 리샘플링 라이브러리(`torchaudio.transforms.Resample` 또는 `librosa.resample`), 안티앨리어싱(anti-aliasing) 필터 유형.
3. 채널 계획(Channel plan): 모노 폴딩(Mono folding) 전략(평균값 사용 vs 왼쪽 채널만 사용), 또는 모델이 지원하는 경우 멀티채널 패스스루(pass-through).
4. 정규화(Normalization): Peak vs RMS 정규화, dBFS 목표값, 클리핑 방지(clipping guard).
5. 검증 스니펫(Validation snippet): 파일을 로드하고, 변환을 실행하며, 최종 배열이 `(target_sr, dtype, channel_count, range)`와 일치하는지 확인(`assert`)하는 Python 코드.

안티앨리어싱 필터 없이 다운샘플링하는 것은 거부하세요. 재구성 필터(reconstruction filter) 없이 2배 이상의 업샘플링을 수행하는 것도 거부하세요. 클리핑 피크가 ±0.999를 초과하거나 DC 오프셋이 ±0.01을 초과하는 입력 파일은 모두 플래그(flag)를 표시하세요.
