# Yacs 환경설정

Detectron2는 일반적인 표준 동작을 얻는 데 사용할 수 있는 키-밸류 기반 환경설정 시스템을 제공합니다.

이 시스템은 YAML 및 [yacs](https://github.com/rbgirshick/yacs) 를 사용합니다.
YAML은 제약이 많은 언어이므로 환경설정을 통해 detectron2의 모든 기능을 사용할 수는 없습니다.
만일 환경설정을 통해 필요한 기능이 제공되지 않는 경우, detectron2의 API를 사용하여 코드를 작성하십시오.

이보다 강력한 [LazyConfig 시스템](lazyconfigs.md) 이 도입되면서, 저희는 더 이상 Yacs/Yaml 기반 환경설정 시스템에 새로운 기능/키를 추가하지 않고 있습니다.

### 기본 사용법

`CfgNode` 객체의 기본적인 사용법은 다음과 같습니다. 자세한 내용은 [문서](../modules/config.html#detectron2.config.CfgNode) 를 참조하십시오.
```python
from detectron2.config import get_cfg
cfg = get_cfg()    # detectron2의 기본 환경설정을 불러옴
cfg.xxx = yyy      # 사용자가 정의한 새 환경변수에 대한 설정값 추가
cfg.merge_from_file("my_cfg.yaml")   # 파일에서 값 로드

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # str list에서 값을 로드할 수도 있음
print(cfg.dump())  # 환경설정을 포매팅하여 출력
with open("output.yaml", "w") as f:
  f.write(cfg.dump())   # 파일에 환경설정 저장
```

기본 Yaml 구문 외에도 기본 환경설정 파일을 먼저 로드하도록
`_BASE_: base.yaml` 필드를 환경설정 파일에 정의할 수 있습니다.
conflict가 발생할 경우, 하위 환경설정이 기본 환경설정의 밸류를 덮어씁니다.
표준 모델 아키텍처의 경우, 몇 가지 환경설정이 기본으로 제공됩니다.

detectron2의 많은 내장 도구는 커맨드라인 명령을 통한 덮어쓰기를 허용하며,
이 때 입력된 키-밸류 쌍은 환경설정 파일의 기존 밸류를 덮어씁니다.
예를 들어, 다음과 같이 [demo.py](../../demo/demo.py) 를 사용할 수 있습니다.
```sh
./demo.py --config-file config.yaml [--other-options] \
  --opts MODEL.WEIGHTS /path/to/weights INPUT.MIN_SIZE_TEST 1000
```

detectron2에서 사용 가능한 환경설정 목록과 그 의미는
[환경설정 참고자료](../modules/config.html#config-references) 에서 확인하십시오.


### 프로젝트 환경설정

detectron2 라이브러리 외부에 있는 프로젝트는 자체 환경설정을 정의할
수 있으며, 프로젝트가 동작하기 위해 이를 추가해야 할 수 있습니다. e.g.:
```python
from detectron2.projects.point_rend import add_pointrend_config
cfg = get_cfg()    # detectron2의 기본 환경설정을 불러옴
add_pointrend_config(cfg)  # pointrend의 기본 환경설정을 추가
# ... ...
```

### 환경설정 잘하는 법

1. "코드"라고 생각하고 작성하기: 환경설정을 복사/복제하지 말고 `_BASE_` 를 사용하여
   여러 환경설정 간 공통 부분을 공유하십시오.

2. 복잡하지 않게 작성하기: 실험에 영향을 주지 않는 키를 포함하지 마십시오.
