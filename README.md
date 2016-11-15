# 딥러닝 기반의 멀티-모달 MRI 영상에서의 뇌종양 영역 분할

***한국정보과학회 제43회 정기총회 및 동계학술발표회 개제용 공개 코드***

## 구현 환경
* Python 2.7.12 (Anaconda Build)
    * 사용 주요 라이브러리: MedPy, Tensorflow
* Ubuntu 14.04
* Hardware
    * CPU: i7-6700K
    * GPU: NVIDIA GTX 970
    * RAM: 16GB

## 필요 사항
1. [BRATS 2015 데이터](https://www.smir.ch/BRATS/Start2015)
2. [Covert3D Tool](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)
3. [3D Slicer](http://download.slicer.org/)
4. 최소 100GB 저장 공간.
    

## 사용 설명

### 다운로드 및 파일 변경

1. BRATS 2015 압축 해제.

    * "input" 폴더 생성 후 test, train 데이터 (HGG, LGG, HGG_LGG) 압축 해제.

2. 불필요 파일 및 폴더 제거 및 파일 이름 변경. ("input" 폴더로 cd 한 후)
```bash
for n in ./*/*/*/*.txt; do rm "$n"; done
for n in ./*/*/*/*; do mv "$n" "$(dirname "$(dirname "$n")")"; done
for n in ./*/*/*/; do rm -r "$n"; done
a=1
cd HGG
for n in *; do  mv "$n" "h.$a";  let a=a+1; done
a=1
cd ../LGG
for n in *; do  mv "$n" "l.$a";  let a=a+1; done
a=1
cd ../HGG_LGG
for n in *; do  mv "$n" "t.$a";  let a=a+1; done
for n in */*.mha; do  new="$(echo "$n" |  sed 's/\//./')"; mv "$n" "$new"; done
cd ../HGG
for n in */*.mha; do  new="$(echo "$n" |  sed 's/\//./')"; mv "$n" "$new"; done
cd ../HGG_LGG
for n in */*.mha; do  new="$(echo "$n" |  sed 's/\//./')"; mv "$n" "$new"; done
cd ..
for n in */*/; do  rm -r "$n"; done
for n in ./*/*; do mv "$n" "$(dirname "$(dirname "$n")")"; done
```

### Metadata (.mha) 파일을 NIfTI (.nii)로 변경 후 N4ITK 툴 실행
1. "input" 폴더가 들어있는 폴더에 "GT", "data" 폴더 생성.
2. "input" 폴더에 "OT" 단어가 포함된 파일 모두 "GT"로 이동.
3. C3D 툴로 .mha 파일을 .nii파일로 변경. ("GT" 파일로 cd 후, c3d 경로 변경 필수)
```bash
for n in *.mha; do   ~/c3d/bin/c3d "./$n" -type uchar ../data/"${n%.mha}.nii"; done
```
4. N4ITK 툴 사용 (3D slicer 경로 변경 필수)
```bash
cd ../input
for n in *.mha; do   ~/slicer/lib/Slicer-4.5/cli-modules/N4ITKBiasFieldCorrection "./$n" ../data/"${n%.mha}.nii"; done
```

### 코드 설명
1. createdb.py 파라미터

    * ROTATE: 90도 회전 횟수. (1->0번, 4->3번)
    * ORIG_READ_PATH: N4ITK가 끝난 .nii 파일들이 들어 있는 폴더이다.
    * WRITE_PATH: 학습 및 실험 중 산출되는 데이터를 저장할 폴더이다.

2. train.py 파라미터

    * NUM_EPOCHS: epoch 횟수.
    * BATCH_SIZE: 각 step 당 데이터 개수.
    * EVAL_FREQUENCY: 몇 step 당 학습 결과를 출력할지.

3. test.py 파라미터

    * VAL_SIZE: 실험할 학습용 데이터 개수.
    
    
## 문의
윤지석
고려대학교 컴퓨터학과
wltjr1007@korea.ac.kr
