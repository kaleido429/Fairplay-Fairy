<h1>주요 변경사항</h1>

<ol>
  <li>
    <strong>명중 감지 마스킹 영역 축소</strong><br>
    명중 시 이펙트와 유사하게 마스킹 영역을 <em>x자 모양</em>으로 좁혔습니다.
  </li>

  <li>
    <strong>마스킹 연산 변경</strong><br>
    기존에 사용하던 <code>cv2.morphologyEx()</code> 대신 <code>cv2.dilate()</code>를 사용하도록 변경했습니다.
    (이전에는 명중 이펙트가 너무 작아 마스킹 영역이 아예 사라지는 문제가 있었습니다; dilate로 해당 문제를 해결)
  </li>

  <li>
    <strong>총기 쿨타임 조정</strong><br>
    쿨타임을 <code>100 → 120</code>으로 상향 조정했습니다.
    (한 번 발사 시 슈팅 횟수가 2~3회 증가하던 현상을 제거하기 위한 변경)
  </li>

  <li>
    <strong>HSV 영역 미세 조정</strong><br>
    색상(HSV) 영역을 약간 수정하여 감지 정확도를 향상시켰습니다.
  </li>
</ol>

