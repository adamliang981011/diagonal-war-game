# 已知問題

## 1. 非遊戲視窗模糊（NVIDIA 筆電 GPU）

- **描述**：開啟遊戲後，所有非遊戲視窗的圖片與文字皆變模糊。
- **原因**：Vulkan swapchain 建立時，NVIDIA 驅動進入「遊戲效能模式」，觸發 DWM
  （Desktop Window Manager）重新合成桌面，導致非遊戲視窗渲染異常。
- **影響範圍**：僅限搭載 NVIDIA Optimus 技術的筆記型電腦。
- **嘗試過的修正**：
  - 改用 DX12 後端（`Backends::DX12`）→ 部分環境改善但仍可能發生
- **待嘗試方案**：
  - NVIDIA 控制面板 → 管理 3D 設定 → 電源管理模式 → 「一般」
  - Windows 圖形設定 → 將 `diagonal-war.exe` 加入 → 選「省電」（強制使用內顯）
  - 若無效，可能需要等待驅動更新或改用其他渲染後端
- **優先級**：低（不影響遊戲功能）

## 2. Entity despawn WARN（scroll container 清除時）

- **描述**：每次換玩家或面板更新時，終端機出現 `WARN bevy_ecs::error::handler: Encountered an error in command ... Entity despawned`。
- **原因**：Bevy 0.19 的 ScrollPosition / Overflow 內部系統在 PostUpdate 階段嘗試操作已被清除的 entity。此為 Bevy 0.19 本身的問題（參見 GitHub Issue #18933）。
- **影響範圍**：無任何遊戲功能、記憶體或效能影響。純 log 噪音。
- **是否可從應用端修正**：否。需在 Bevy 後續版本修正。應用端的 `EntityCommands::despawn()` 已使用 `queue_silenced` 內部抑制，但 Bevy 內部其他系統仍會發出此 WARN。
- **優先級**：極低（僅 log 噪音，不影響遊玩）
