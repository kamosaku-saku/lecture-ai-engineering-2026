# 第1回 解答

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：
- 用語集の正しい意味と比べて気づいたこと：
- 表示できたトークン数（入力 / 出力）：

## 宿題

- 取り組んだレベル：基礎
- やったこと：モデル、temperatureの違いによる出力の比較を行った。(gemini-3.5-flash-lite, temperature=0.0)、(gemini-3.5-flash-lite, temperature=0.7)、(gemini-3.5-flash, temperature=0.7)の三パターンで試した。
- 結果：temperatureが高いほど、砕けた表現（「すごい」など）を使う傾向があった。また、gemini-3.5-flashは推論時間がgemini-3.5-flash-liteよりも長く、出力もより詳細なものになった。
- 気づき・疑問：モデル、temperatureの違いが出力の質に違いを生む。
