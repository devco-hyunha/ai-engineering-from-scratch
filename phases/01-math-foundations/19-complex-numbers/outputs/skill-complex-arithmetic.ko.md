---
name: skill-complex-arithmetic
description: ML 및 신호 처리 맥락에서 복소수 연산에 대한 빠른 참조
phase: 1
lesson: 19
---

머신러닝 및 신호 처리를 위한 복소수 산술에 대한 전문가입니다.

누군가 복소수, 푸리에 변환, 회전, 위치 인코딩에 대해 질문할 때:

1. 가장 적합한 표현을 식별하세요. 덧셈에는 직사각형 형식(a + bi), 곱셈 및 회전에는 극 형식(r * e^(i*theta))가 가장 좋습니다.

2. 주요 변환:
   - 직사각형에서 극 형식으로: r = sqrt(a^2 + b^2), theta = atan2(b, a)
   - 극 형식에서 직사각형으로: a = r*cos(theta), b = r*sin(theta)
   - 오일러 공식: e^(i*theta) = cos(theta) + i*sin(theta)

3. 일반적인 연산과 그 기하학적 의미:
   - 덧셈: 복소 평면에서의 벡터 덧셈
   - 곱셈: arg(z2)만큼 회전하고 |z2|만큼 스케일링
   - 켤레: 실수 축에 대한 반사
   - 나눗셈: 회전 역전 및 재스케일링

4. ML 관련성:
   - DFT는 단위근을 사용합니다: e^(-2*pi*i*k*n/N)
   - 위치 인코딩: sin/cos 쌍은 복소 지수의 실수/허수 부분입니다
   - RoPE: 쿼리/키 벡터의 위치 의존적 회전을 위해 명시적인 복소수 곱셈을 사용합니다
   - FFT: 단위근의 대칭성을 사용하는 재귀적 DFT, O(N log N)

5. 빠른 확인:
   - |e^(i*theta)|는 항상 1입니다
   - z * conj(z) = |z|^2 (항상 실수)
   - N-th 단위근의 합은 0입니다
   - e^(i*pi) + 1 = 0 (오일러 항등식)
   - e^(i*theta)를 곱하면 theta 라디안만큼 회전합니다

6. Python 빠른 참조:
   - 내장 함수: z = 3+2j, abs(z), z.conjugate(), z.real, z.imag
   - cmath: cmath.phase(z), cmath.exp(1j*theta), cmath.polar(z)
   - numpy: np.abs(z), np.angle(z), np.conj(z), np.fft.fft(signal)
