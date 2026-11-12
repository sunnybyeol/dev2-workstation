# 화면 증거

이 폴더는 실제 앱 화면 캡처를 보관하는 위치입니다. 이 실행 환경에서는 VS Code 창을 대상으로 한 화면 캡처를 저장할 수 없어 이미지를 임의로 생성하지 않았습니다. 재현 가능한 실제 실행 출력은 `../logs/`에 저장했습니다.

- `../logs/menu-exit.txt`: 시작 메뉴와 종료
- `../logs/mode-json.txt`: 9개 데이터 사례의 PASS 결과 및 성능 측정
- `../logs/bonus-generator.txt`: 자동 패턴 생성 및 2D/1D 비교
- `../logs/tests.txt`: 표준 라이브러리 단위 테스트 결과

직접 화면 캡처를 추가할 때는 Python 프로그램의 메뉴/분석 화면을 띄운 뒤 Windows 캡처 도구로 저장하고, 이 폴더에 `menu.png`, `json-results.png`, `bonus.png` 이름으로 넣으면 됩니다.
