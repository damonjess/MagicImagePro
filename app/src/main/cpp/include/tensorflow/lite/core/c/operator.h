#ifndef TENSORFLOW_LITE_CORE_C_OPERATOR_H_
#define TENSORFLOW_LITE_CORE_C_OPERATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TfLiteOperator TfLiteOperator;

inline TfLiteOperator* TfLiteOperatorCreate() { return nullptr; }
inline void TfLiteOperatorDelete(TfLiteOperator* op) {}
inline int TfLiteOperatorGetBuiltInCode(const TfLiteOperator* op) { return 0; }
inline int TfLiteOperatorGetVersion(const TfLiteOperator* op) { return 1; }
inline void TfLiteOperatorSetInit(TfLiteOperator* op, void* init) {}
inline void TfLiteOperatorSetFree(TfLiteOperator* op, void* free) {}
inline void TfLiteOperatorSetPrepare(TfLiteOperator* op, void* prepare) {}
inline void TfLiteOperatorSetInvoke(TfLiteOperator* op, void* invoke) {}
inline const char* TfLiteOperatorGetCustomName(const TfLiteOperator* op) { return ""; }

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_CORE_C_OPERATOR_H_
