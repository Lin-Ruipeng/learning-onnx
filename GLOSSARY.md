# ONNX Deployment Learning Repository

This repository preserves an earlier deployment course and develops a separately maintained modern ONNX course. The terms below distinguish historical material from the actively verified learning path.

## Language

**Historical Course Archive**:
A frozen, independently understandable record of the completed MMDeploy-based learning course, including its code, dependencies, assets, unfinished items, and migration notes.
_Avoid_: Legacy code, deprecated tutorial, old files

**Modern ONNX Course**:
The actively maintained, hands-on course for learning current ONNX export, validation, transformation, optimization, and deployment practices.
_Avoid_: New folder, replacement tutorial

**Living Course**:
A course whose explanations are backed by executable examples, automated correctness checks, locked dependencies, and recorded compatibility expectations so that future upgrades can be verified.
_Avoid_: Documentation collection, article series

## Course Structure

**Core Course**:
The hardware-independent, required learning path that establishes modern ONNX export, validation, transformation, optimization, and deployment fundamentals.
_Avoid_: Basic course, beginner course

**Advanced Graph Engineering**:
The advanced course layer concerned with ONNX graph semantics, safe transformations, custom translations, diagnostic models, and backend compatibility analysis.
_Avoid_: ONNX tricks, graph hacks

**Backend Track**:
An optional, independently runnable course path for a specific runtime, accelerator, or device platform, built on the Core Course.
_Avoid_: Platform appendix, vendor chapter

**TCN Deployment Case Study**:
The recurring real-world case study that connects model export, artifact validation, graph transformation, backend conversion, device execution, and deployment acceptance.
_Avoid_: TCN example, extra demo

**SRCNN Migration Bridge**:
The bounded comparison case that connects lessons from the Historical Course Archive to the current exporter and runtime practices taught by the Modern ONNX Course.
_Avoid_: Main course model, legacy example

**Teaching Fixture**:
A minimal, provenance-recorded model, graph, dataset slice, or expected result extracted from a source project to teach and test one bounded lesson independently.
_Avoid_: Project copy, sample file

**Course Environment**:
A reproducible dependency boundary owned by one course context or vendor-constrained Backend Track.
_Avoid_: Shared global environment, repository environment

## Graph Engineering

**Graph Transformation**:
A deliberate change to an ONNX graph performed under explicit structural and semantic preconditions.
_Avoid_: ONNX editing trick, graph hack

**Production Rewrite**:
A Graph Transformation intended for deployable artifacts that must preserve the declared model semantics and pass equivalence validation.
_Avoid_: Optimization trick, safe edit

**Diagnostic Mutation**:
A Graph Transformation that intentionally changes semantics to isolate a failure and is never treated as a deployable equivalent of the source model.
_Avoid_: Production rewrite, approximate fix

**Exporter Translation**:
The export-side mapping that expresses a framework operation as standard or custom ONNX graph semantics; it does not provide a runtime implementation.
_Avoid_: Custom operator implementation, symbolic hack

**Runtime Custom Operator**:
The runtime-side implementation that executes an operator not provided by the runtime's standard operator set; it does not define how a framework model is exported.
_Avoid_: Exporter translation, graph rewrite

**Derived Artifact**:
A newly generated model artifact produced from an immutable source artifact by an explicitly recorded conversion or Graph Transformation.
_Avoid_: Edited source model, overwritten model

## Verification

**Artifact Contract**:
The declared identity, inputs, outputs, data types, shapes, semantics, and compatibility expectations that a model artifact must satisfy independently of whether a runtime can load it.
_Avoid_: Model signature, successful export

**Verified Baseline**:
A locked toolchain and execution path whose correctness tests have passed and which serves as the reproducible reference for a course chapter or Backend Track.
_Avoid_: Latest version, working environment

**Artifact Manifest**:
The identity and provenance record for a generated model artifact, including the source, interface contract, toolchain, and integrity information needed to reproduce and validate it.
_Avoid_: Model notes, version comment

**Hardware Acceptance Report**:
The recorded correctness and performance result produced by running a Conditional Backend Lab on the intended physical device.
_Avoid_: Benchmark screenshot, manual note

**Conditional Backend Lab**:
A Backend Track experiment that automatically checks required capabilities, skips cleanly when they are absent, and is accepted only through a Hardware Acceptance Report from suitable hardware.
_Avoid_: Optional untested code, hardware demo

**RK3588 Backend Track**:
The first device-specific learning path, targeting Linux on RK3588 and covering model preparation, device execution, quantization, and performance engineering.
_Avoid_: Android track, RKNN demo

**Toolchain Smoke Model**:
A small model with known backend support used to prove that conversion and device execution work before compatibility is evaluated with the TCN Deployment Case Study.
_Avoid_: Toy benchmark, final case study

**Golden Reference**:
The trusted output produced by a verified source-model or portable-runtime path against which transformed and device-executed artifacts are compared.
_Avoid_: Expected-looking output, simulator result

**Device Profile**:
The recorded identity and compatible software stack of a deployment device, including its operating system, kernel, accelerator driver, runtime, and reported hardware capabilities.
_Avoid_: Board setup, environment notes

**Four-Way Validation**:
The comparison of source-model, portable-runtime, vendor-preparation, and physical-device results against one declared Artifact Contract and input corpus.
_Avoid_: Simulator validation, device smoke test
