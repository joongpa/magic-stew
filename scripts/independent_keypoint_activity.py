"""One-frame activity evidence in the physical state coordinate system."""
import math


UNOBSERVED = 'unobserved'


def layer(value):
    """Pose-rescued trials already express these axes in physical coordinates."""
    if all(value.get(k) is not None for k in ('layer_axis', 'layer_side', 'layer_count')):
        return tuple(int(value[k]) for k in ('layer_axis', 'layer_side', 'layer_count'))
    return None


def observed_layers(payload):
    """A move or any nonzero partial is activity; direction is immaterial."""
    result = set()
    if payload.get('state_moves') and layer(payload) is not None:
        result.add(layer(payload))
    for rotation in payload.get('layer_rotations', ()):
        angle = float(rotation['angle_degrees'])
        if math.isfinite(angle) and abs(angle) > 1e-8:
            result.add(tuple(int(rotation[k]) for k in ('axis', 'side', 'layer_count')))
    return tuple(sorted(result))


def previous(step, frame):
    """Evidence expires immediately: only frame-1 can support a move."""
    if step.get('frame_index') != frame-1:
        return ()
    if step.get('bridge_inferred'):
        # A globally unobserved frame supplies no evidence to rule out any
        # layer. Preserve the existing blind-frame bridge policy explicitly.
        return UNOBSERVED
    return observed_layers(step)


def permits(evidence, payload):
    if not payload.get('state_moves') or evidence == UNOBSERVED:
        return True
    return layer(payload) in evidence


def log_fields(evidence, payload):
    return dict(previous_layer_activity=evidence,
                layer_activity=observed_layers(payload),
                activity_evidence_status=('unobserved_previous_frame' if evidence == UNOBSERVED
                                          else 'observed_previous_frame'))
