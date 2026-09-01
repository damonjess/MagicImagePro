# Integrate ObjectSnapper for Smart Select

This plan integrates the `ObjectSnapper` class (Segment Anything Model) into the app, allowing users to select objects with a single tap.

## User Review Required

> [!IMPORTANT]
> A Segment Anything Model (SAM) TFLite file (e.g., `edgesam.tflite`) is required in the `assets` folder for this feature to work. I will assume the filename is `edgesam.tflite` in the code, but you can change it if your model has a different name.

## Proposed Changes

### UI & UX

#### [MODIFY] [MaskDrawingView.kt](file:///C:/Users/Damon/AndroidStudioProjects/MagicImagePro/app/src/main/java/com/example/magicimagepro/ui/MaskDrawingView.kt)
- Add an `EditMode` enum (BRUSH, SMART_SELECT).
- Add a listener interface/callback for when a tap occurs in `SMART_SELECT` mode.
- Update `onTouchEvent` to trigger the callback on `ACTION_UP` when in `SMART_SELECT` mode.
- Add `setMask(mask: Bitmap)` method to update the mask from an external source (like `ObjectSnapper`).

#### [MODIFY] [MainActivity.kt](file:///C:/Users/Damon/AndroidStudioProjects/MagicImagePro/app/src/main/java/com/example/magicimagepro/MainActivity.kt)
- Add `ObjectSnapper` as a class property.
- Initialize `ObjectSnapper` in `onCreate`.
- Update `setupButtons` to toggle between Brush and Smart Select modes.
- Implement the `onTap` listener from `MaskDrawingView` to call `ObjectSnapper.generateMask()` and update the UI.

## Verification Plan

### Manual Verification
- Deploy the app to a device/emulator.
- Load an image.
- Toggle to "Smart Select" mode (by clicking the mode button).
- Tap on an object in the image.
- Verify that a mask is generated automatically around the object.
- Use the "Erase Object" button to remove the snapped object.
