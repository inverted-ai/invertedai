# Environment Variables



|   name   | default | options  | description |
|:-----------:|:-----------:|:-----------:|:-----------:|
| IAI_LOG_LEVEL | `WARNING` | [`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`]| Supported python logger levels| 
|   IAI_LOG_CONSOLE    |  `true`   |  [`y`, `yes`, `t`, `true`, `on`, `1`, `n`, `no`, `f`, `false`, `off`, `0`]| Whether to log to the console|
|    IAI_LOG_FILE    |  `false`   | [`y`, `yes`, `t`, `true`, `on`, `1`, `n`, `no`, `f`, `false`, `off`, `0`] | Whether to log to the file `iai.log`|
 |     IAI_API_KEY     |    `""`    | NA | API Key needed to call the InvertedAI API|
 |     IAI_MOCK_API     |    `false    | [`y`, `yes`, `t`, `true`, `on`, `1`, `n`, `no`, `f`, `false`, `off`, `0`] | If true it will call the Mock API instead|