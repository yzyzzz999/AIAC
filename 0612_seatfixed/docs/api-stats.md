# `/stats` 接口说明

部署模式下仅 4 个端点可用：`/health` `/stats` `POST /recognition/stop` `POST /recognition/resume`

---

## 响应示例

```json
{
  "status": "ok",
  "timestamp": "2026-06-18T10:30:12.123456",
  "running": true,
  "paused": false,
  "frame_count": 12345,
  "total_detected": 2,
  "total_known": 1,
  "gallery_count": 5,
  "driver": {
    "identity_id": "a1b2c3d4e5f67890",
    "gender": 1,
    "age": 3,
    "cloth": 0,
    "height": 175.2,
    "bmi": 22.5
  },
  "passenger": null,
  "enum_map": {
    "gender": {"0": "female", "1": "male"},
    "age": {"0": "12-17", "1": "18-40", "2": "41-59", "3": "60-74", "4": "75+"},
    "cloth": {
      "0": "西装外套", "1": "薄夹克", "2": "长款大衣", "3": "羽绒服",
      "4": "长袖针织毛衣", "5": "长袖衬衫", "6": "短袖", "7": "背心",
      "8": "长袖", "9": "连帽卫衣"
    }
  }
}
```

## 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` 表示正常 |
| `timestamp` | string | ISO 8601 时间戳 |
| `running` | bool | 识别线程是否在运行 |
| `paused` | bool | 识别是否已暂停 |
| `frame_count` | int | 累计处理帧数 |
| `total_detected` | int | 当前帧检测到的人脸数 |
| `total_known` | int | 当前帧已识别的人脸数 |
| `gallery_count` | int | 已注册身份总数 |
| `driver` | object\|null | 主驾信息，无人时为 `null` |
| `passenger` | object\|null | 副驾信息，无人时为 `null` |
| `enum_map` | object | 枚举值对照表，供调用方解析数字字段 |

### driver / passenger 子字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `identity_id` | string | 身份唯一标识（16 位 hex，Gallery 注册 ID） |
| `gender` | int\|null | 0=female, 1=male，未就绪时为 `null` |
| `age` | int\|null | 年龄段枚举，见下表，未就绪时为 `null` |
| `cloth` | int\|null | 衣着枚举，见下表，未就绪时为 `null` |
| `height` | float\|null | 身高估计值（cm），未就绪时为 `null` |
| `bmi` | float\|null | BMI 值，未就绪时为 `null` |

---

## 枚举对照表

### gender

| 值 | 含义 |
|----|------|
| 0 | female |
| 1 | male |

### age

| 值 | 含义 |
|----|------|
| 0 | 12-17 |
| 1 | 18-40 |
| 2 | 41-59 |
| 3 | 60-74 |
| 4 | 75+ |

### cloth

| 值 | 含义 |
|----|------|
| 0 | 西装外套 |
| 1 | 薄夹克 |
| 2 | 长款大衣 |
| 3 | 羽绒服 |
| 4 | 长袖针织毛衣 |
| 5 | 长袖衬衫 |
| 6 | 短袖 |
| 7 | 背心 |
| 8 | 长袖 |
| 9 | 连帽卫衣 |

---

## 行为说明

- **座位状态保持**：人脸短暂丢失（如眨眼）时，保持最近 3 秒内的有效数据，不会立即返回 `null`
- **逐字段合并**：各属性（性别、年龄、衣着、身高、BMI）就绪时间不同，就绪的字段用当前值、未就绪的沿用上次有效值，保证每次返回的都是当前最完整的数据
- **换人清空**：检测到 `identity_id` 变化时立即清空旧数据
- `height` 和 `bmi` 是连续值，不在 `enum_map` 中
- 座位无人时对应位置整体为 `null`
